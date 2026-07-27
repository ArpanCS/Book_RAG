import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from generate import generate_answer
from search import EMBEDDING_MODEL_NAME, build_bm25_index, get_collection, hybrid_search

load_dotenv()

# Edit this list to try your own questions.
QUESTIONS = [
    "What is the name of the aunt who raises Harry?",
    "What Gringotts vault number does Hagrid collect a package from?",
    "What is the name of Neville's toad?",
    "Why couldn't the Dursleys leave Harry with Mrs. Figg before their zoo trip?",
    "How many points did Neville earn at the final feast, and why?",
    "Why did Voldemort want the Sorcerer's Stone so badly?",
    "Why does Quirrell wear a turban all year, and what is the payoff of that detail?",
]

N_RESULTS = 15


def main():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = get_collection()
    bm25_index = build_bm25_index(collection)  # built once, reused for every question
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    for question in QUESTIONS:
        results = hybrid_search(question, model, collection, bm25_index, n_results=N_RESULTS)

        print("=" * 70)
        print(f"Question: {question}\n")

        print(f"Top {N_RESULTS} chunks retrieved (hybrid: embeddings + BM25):\n")
        for rank, (cid, doc, meta) in enumerate(
            zip(results["ids"], results["documents"], results["metadatas"]),
            start=1,
        ):
            print(f"  #{rank}  {cid}  ({meta['chapter']})")
            print(f"  {doc}\n")

        answer = generate_answer(question, results["documents"], groq_client)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
