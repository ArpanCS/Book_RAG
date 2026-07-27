import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from search import EMBEDDING_MODEL_NAME, build_bm25_index, get_collection, hybrid_search

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# This system message is the entire "only answer from this book" guardrail --
# there's no special filter or setting elsewhere, it's just these instructions.
SYSTEM_INSTRUCTIONS = """You are answering questions about the book "Harry Potter and the Sorcerer's Stone" ONLY.
Use ONLY the context the user gives you to answer. Do not use any knowledge about
other Harry Potter books, movies, or anything outside the given context.
If the answer is not contained in the context, say: "I don't have enough
information in this book to answer that." """


def build_user_message(question, chunks):
    context = "\n\n".join(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question, chunks, groq_client, model=GROQ_MODEL):
    user_message = build_user_message(question, chunks)
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = get_collection()
    bm25_index = build_bm25_index(collection)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    questions = [
        "What house does Harry get sorted into?",  # in-book, but hybrid search (n_results=5) still misses the actual sorting-scene chunks (chunk_144/145)
        "What happens in Harry Potter and the Chamber of Secrets?",  # out-of-book series question
    ]

    for question in questions:
        results = hybrid_search(question, model, collection, bm25_index, n_results=5)
        answer = generate_answer(question, results["documents"], groq_client)

        print("=" * 70)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")
