"""
LLM-as-judge: faithfulness check.

Faithfulness asks a different question than the semantic-similarity check in
evaluate.py. Similarity asks "does this answer SOUND like the truth?" --
faithfulness asks "is every claim in this answer actually SUPPORTED by the
retrieved context, or did the model make something up?" A confidently wrong
number (e.g. "717" instead of "713") can score fine on similarity but should
fail faithfulness, because "717" never appears in the context at all.

This uses a second, separate LLM call -- a "judge" -- whose only job is to
fact-check the generated answer against the context it was given. It is not
judging writing quality or completeness, only truthfulness relative to the
context.
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# openai/gpt-oss-120b, not llama-3.3-70b-versatile: on 2026-07-28 the smaller
# llama-3.1-8b-instant judge missed a real hallucination (said a wrong vault
# number was "faithful" to context that actually had a different number).
# gpt-oss-120b correctly caught that same case, and hasn't hit Groq's daily
# token quota the way llama-3.3-70b-versatile repeatedly has -- see
# pipeline 3/evaluation_dataset.txt-adjacent memory for the full story.
JUDGE_GROQ_MODEL = "openai/gpt-oss-120b"

JUDGE_SYSTEM_INSTRUCTIONS = """You are a strict fact-checker. You will be given a QUESTION, some CONTEXT
passages from a book, and an ANSWER that was generated using that context.

Your only job is to check whether every factual claim in the ANSWER is actually
supported by the CONTEXT -- names, numbers, and specific details must genuinely
appear in the CONTEXT, not just be plausible-sounding. Do not judge whether the
answer is well-written, complete, or matches some outside idea of the "correct"
answer -- only whether its claims are true according to the CONTEXT given.

Respond in EXACTLY this format, nothing else:
FAITHFUL: yes or no
REASON: one short sentence. If not faithful, name the specific unsupported
claim (e.g. a wrong number, name, or fact that doesn't match the context)."""


def build_judge_message(question, context_chunks, answer):
    context = "\n\n".join(context_chunks)
    return f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"


def judge_faithfulness(question, context_chunks, answer, groq_client, model=JUDGE_GROQ_MODEL):
    user_message = build_judge_message(question, context_chunks, answer)
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ],
    )
    verdict_text = response.choices[0].message.content

    faithful = None
    reason = verdict_text.strip()
    for line in verdict_text.splitlines():
        line = line.strip()
        if line.upper().startswith("FAITHFUL:"):
            faithful = "yes" in line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {"faithful": faithful, "reason": reason, "raw": verdict_text}


if __name__ == "__main__":
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Two known real cases from the 2026-07-28 evaluation run, used here
    # without re-running retrieval/generation, just to sanity-check the judge
    # itself before wiring it into evaluate.py.
    test_cases = [
        {
            "label": "known hallucination (should be judged NOT faithful)",
            "question": "What Gringotts vault number does Hagrid collect a package from?",
            "context_chunks": [
                "Hagrid pulled a grubby little package out of an inside pocket. "
                "\"Got something for you an' all,\" he says. \"Vault seven hundred "
                "and thirteen. Do you want to come with me while I get it? And -- "
                "yeah, you might as well see the bank while you're there.\""
            ],
            "answer": "717",
        },
        {
            "label": "known correct answer (should be judged faithful)",
            "question": "What is the name of the goblin who leads Harry and Hagrid to the vaults?",
            "context_chunks": [
                "\"Follow me,\" said Griphook. Harry followed Griphook, who was "
                "leading them toward one of the doors leading off the hall."
            ],
            "answer": "The name of the goblin who leads Harry and Hagrid to the vaults is Griphook.",
        },
    ]

    for case in test_cases:
        verdict = judge_faithfulness(
            case["question"], case["context_chunks"], case["answer"], groq_client
        )
        print("=" * 70)
        print(case["label"])
        print("Question:", case["question"])
        print("Answer:  ", case["answer"])
        print("Verdict: FAITHFUL =", verdict["faithful"])
        print("Reason:  ", verdict["reason"])
        print()
