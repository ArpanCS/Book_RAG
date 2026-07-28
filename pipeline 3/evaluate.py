"""
Pipeline 3: evaluation.

Three separate checks, run against every question in evaluation_dataset.txt:

1. RETRIEVAL evaluation (Set 1 / easy questions only, for now): did hybrid
   search actually put the right chunk inside the N_RESULTS we hand to the
   LLM? Checked by looking for a hand-verified, distinctive keyword phrase
   from the true answer inside the retrieved chunks' text.

2. GENERATION evaluation (all 60 questions): how close in MEANING is the
   LLM's generated answer to the ground-truth answer? Measured with
   cosine similarity between the two answers' embeddings (same idea as
   comparing chunk vectors, just applied to answers instead).

3. FAITHFULNESS evaluation (all 60 questions, via judge.py): is every claim
   in the generated answer actually supported by the retrieved context, or
   did the model make something up? This catches confidently-wrong answers
   that similarity alone can miss (e.g. a hallucinated number that "sounds"
   right but isn't in the context).
"""
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv
from groq import Groq, RateLimitError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline 2"))
from generate import generate_answer  # noqa: E402
from search import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    RERANK_MODEL_NAME,
    build_bm25_index,
    get_collection,
    hybrid_search_reranked,
)
from judge import judge_faithfulness  # noqa: E402

from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

DATASET_PATH = os.path.join(os.path.dirname(__file__), "evaluation_dataset.txt")
N_CANDIDATES = 30  # wide shortlist pulled by hybrid search, before reranking
N_RESULTS = 15  # final chunk count kept after reranking -- must match pipeline 2/test_questions.py

# openai/gpt-oss-120b: on 2026-07-28, llama-3.1-8b-instant was used here only
# as a quota workaround, but repeatedly failed reasoning tests (refusing
# answers it had the facts for) and even failed as the judge model (missed a
# real hallucination). llama-3.3-70b-versatile is the production model but
# kept hitting Groq's 100,000 tokens/day cap mid-run. gpt-oss-120b passed
# every retest that day and, as a separate model, has its own separate quota.
EVAL_GROQ_MODEL = "openai/gpt-oss-120b"
# 5 per tier (15 total), not the full 20/tier (60 total): each question now
# costs ~10,000-12,000 tokens (generate + judge calls, each stuffing 15
# chunks into the prompt), and gpt-oss-120b's daily cap is 200,000 tokens.
# 15 questions leaves a real buffer for retries/other testing that day; the
# full 60 would need to be split across multiple days regardless of model.
QUESTIONS_PER_SET = 5

# Hand-verified keyword needles for Set 1 (easy) questions only, in order.
# Each needle is a distinctive phrase that ONLY appears in the chunk(s) that
# actually answer the question -- used to check "did retrieval find it."
SET1_KEYWORDS = [
    "aunt petunia",
    "grunnings",
    "made drills",
    "hedwig",
    "trevor",
    "leaky cauldron",
    "ollivander",
    "phoenix feather",
    "seven hundred and thirteen",
    "griphook",
    "fluffy",
    "norbert",
    "smeltings",
    "figg",
    "pig snout",
    "invisibility cloak",
    "nimbus two thousand",
    "seeker",
    "charlie",
    "flamel",
]


def parse_dataset(path):
    """
    Reads evaluation_dataset.txt and returns a list of
    {"set": "EASY"/"MEDIUM"/"HARD", "question": ..., "answer": ...} dicts.
    """
    text = open(path, encoding="utf-8").read()

    set_blocks = re.split(r"SET \d+ -- (EASY|MEDIUM|HARD / DERIVED)[^\n]*\n=+\n", text)[1:]
    # set_blocks alternates: [set_name, block_text, set_name, block_text, ...]

    entries = []
    for set_name, block in zip(set_blocks[0::2], set_blocks[1::2]):
        set_name = "HARD" if set_name.startswith("HARD") else set_name
        # Each question looks like:  "N. Q: ...\n   A: ... (possibly wrapped\n   across lines)"
        for match in re.finditer(
            r"\d+\.\s*Q:\s*(.+?)\n\s*A:\s*(.+?)(?=\n\n|\Z)", block, re.DOTALL
        ):
            question = " ".join(match.group(1).split())
            answer = " ".join(match.group(2).split())
            entries.append({"set": set_name, "question": question, "answer": answer})
    return entries


def cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def call_with_retry(fn, *args, max_retries=6, wait_seconds=20, **kwargs):
    """
    openai/gpt-oss-120b has a tokens-PER-MINUTE cap (not a daily cap like
    llama-3.3-70b-versatile), so a 429 here just means "wait a bit," not
    "give up for the day." Retries with a fixed wait, since the limit is a
    60-second rolling window.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            print(f"    (rate limited, waiting {wait_seconds}s before retry: {e})", flush=True)
            time.sleep(wait_seconds)


def evaluate_retrieval(easy_entries, model, collection, bm25_index, cross_encoder):
    print("=" * 70)
    print(f"RETRIEVAL EVALUATION (Set 1 / easy, {len(easy_entries)} questions)")
    print("=" * 70)

    hits = 0
    for entry, needle in zip(easy_entries, SET1_KEYWORDS):
        results = hybrid_search_reranked(
            entry["question"], model, collection, bm25_index, cross_encoder,
            n_candidates=N_CANDIDATES, n_results=N_RESULTS,
        )
        combined_text = " ".join(results["documents"]).lower()
        hit = needle in combined_text
        hits += hit
        status = "HIT " if hit else "MISS"
        print(f"  [{status}] {entry['question']}")

    print(f"\nRetrieval hit rate: {hits}/{len(easy_entries)}\n")
    return hits, len(easy_entries)


def evaluate_generation(all_entries, model, collection, bm25_index, cross_encoder, groq_client):
    print("=" * 70)
    print(f"GENERATION EVALUATION ({len(all_entries)} questions, similarity + faithfulness)")
    print("=" * 70)

    results_by_set = {"EASY": [], "MEDIUM": [], "HARD": []}
    faithful_by_set = {"EASY": [], "MEDIUM": [], "HARD": []}

    for entry in all_entries:
        retrieved = hybrid_search_reranked(
            entry["question"], model, collection, bm25_index, cross_encoder,
            n_candidates=N_CANDIDATES, n_results=N_RESULTS,
        )
        generated_answer = call_with_retry(
            generate_answer, entry["question"], retrieved["documents"], groq_client, model=EVAL_GROQ_MODEL
        )

        generated_vec = model.encode(generated_answer)
        truth_vec = model.encode(entry["answer"])
        similarity = cosine_similarity(generated_vec, truth_vec)

        verdict = call_with_retry(
            judge_faithfulness, entry["question"], retrieved["documents"], generated_answer,
            groq_client, model=EVAL_GROQ_MODEL,
        )

        results_by_set[entry["set"]].append(similarity)
        faithful_by_set[entry["set"]].append(verdict["faithful"])

        faithful_tag = "FAITHFUL" if verdict["faithful"] else "NOT FAITHFUL"
        print(f"  [{similarity:.3f}] [{faithful_tag}] ({entry['set']}) {entry['question']}")
        print(f"          generated: {generated_answer[:120]}")
        print(f"          truth:     {entry['answer'][:120]}")
        if not verdict["faithful"]:
            print(f"          judge:     {verdict['reason']}")
        print()

    print("-" * 70)
    all_scores = []
    for set_name, scores in results_by_set.items():
        if scores:
            avg = sum(scores) / len(scores)
            faithful_count = sum(1 for f in faithful_by_set[set_name] if f)
            print(
                f"{set_name:<8} avg similarity = {avg:.3f}   "
                f"faithful = {faithful_count}/{len(scores)}  (n={len(scores)})"
            )
            all_scores.extend(scores)
    all_faithful = [f for scores in faithful_by_set.values() for f in scores]
    total_faithful = sum(1 for f in all_faithful if f)
    print(
        f"{'OVERALL':<8} avg similarity = {sum(all_scores)/len(all_scores):.3f}   "
        f"faithful = {total_faithful}/{len(all_faithful)}  (n={len(all_scores)})"
    )

    return results_by_set, faithful_by_set


def main():
    entries = parse_dataset(DATASET_PATH)
    easy_entries = [e for e in entries if e["set"] == "EASY"]
    print(f"Loaded {len(entries)} questions ({len(easy_entries)} easy, "
          f"{sum(e['set']=='MEDIUM' for e in entries)} medium, "
          f"{sum(e['set']=='HARD' for e in entries)} hard).\n")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
    collection = get_collection()
    bm25_index = build_bm25_index(collection)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    evaluate_retrieval(easy_entries, model, collection, bm25_index, cross_encoder)  # no Groq calls, keep all 20

    # Generation eval uses Groq, so trim to QUESTIONS_PER_SET per difficulty tier.
    trimmed = []
    for set_name in ("EASY", "MEDIUM", "HARD"):
        trimmed.extend([e for e in entries if e["set"] == set_name][:QUESTIONS_PER_SET])
    evaluate_generation(trimmed, model, collection, bm25_index, cross_encoder, groq_client)


if __name__ == "__main__":
    main()
