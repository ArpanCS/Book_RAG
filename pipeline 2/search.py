import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"  # must match the model used to build chroma_db
DB_PATH = "chroma_db"
COLLECTION_NAME = "sorcerers_stone_chunks"

RRF_K = 60  # standard damping constant for Reciprocal Rank Fusion


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def _tokenize(text):
    # \w+ pulls out runs of letters/digits, dropping punctuation like ? , . '
    return re.findall(r"\w+", text.lower())


def build_bm25_index(collection):
    """
    Pulls every chunk out of ChromaDB once and builds a BM25 index over the
    raw text. BM25 has nothing to do with vectors -- it just needs each
    chunk's words -- so this is a completely separate index from the one
    ChromaDB uses, kept in memory alongside it.
    """
    data = collection.get(include=["documents", "metadatas"])
    return {
        "ids": data["ids"],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
        "bm25": BM25Okapi([_tokenize(doc) for doc in data["documents"]]),
    }


def search(question, model, collection, n_results=5):
    """
    Embeds a question with the SAME model used for the book chunks, then asks
    ChromaDB for the n_results stored chunks whose vectors are closest to it.

    We pass query_embeddings (a vector we made ourselves), never query_texts --
    query_texts would make ChromaDB embed the question with its own default
    model, which would silently mismatch the model used for the stored chunks.
    """
    query_vector = model.encode(question)
    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    return {
        "ids": results["ids"][0],
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


def hybrid_search(question, model, collection, bm25_index, n_results=5):
    """
    Combines embedding search and BM25 using Reciprocal Rank Fusion (RRF):
    each chunk gets a score of 1/(RRF_K + rank) from EACH method, and the two
    scores are added. We fuse ranks, not raw scores, because BM25 scores and
    embedding distances live on incompatible scales -- rank position ("this
    chunk came in 3rd") is the one thing both methods can be fairly compared
    on.
    """
    all_ids = bm25_index["ids"]
    n_total = len(all_ids)

    # Embedding ranking: ask ChromaDB to rank every chunk, best first.
    query_vector = model.encode(question)
    embed_results = collection.query(query_embeddings=[query_vector], n_results=n_total)
    embed_rank = {cid: rank for rank, cid in enumerate(embed_results["ids"][0], start=1)}

    # BM25 ranking: score every chunk against the tokenized question.
    bm25_scores = bm25_index["bm25"].get_scores(_tokenize(question))
    bm25_ranked_ids = [cid for cid, _ in sorted(zip(all_ids, bm25_scores), key=lambda x: x[1], reverse=True)]
    bm25_rank = {cid: rank for rank, cid in enumerate(bm25_ranked_ids, start=1)}

    # Fuse: add the two per-method reciprocal-rank scores for each chunk.
    fused = sorted(
        all_ids,
        key=lambda cid: 1 / (RRF_K + embed_rank[cid]) + 1 / (RRF_K + bm25_rank[cid]),
        reverse=True,
    )
    top_ids = fused[:n_results]

    id_to_doc = dict(zip(bm25_index["ids"], bm25_index["documents"]))
    id_to_meta = dict(zip(bm25_index["ids"], bm25_index["metadatas"]))
    return {
        "ids": top_ids,
        "documents": [id_to_doc[cid] for cid in top_ids],
        "metadatas": [id_to_meta[cid] for cid in top_ids],
    }


if __name__ == "__main__":
    # Demos the current default (hybrid_search). The older, embeddings-only
    # search() is still defined above and importable for comparison/reference,
    # it's just no longer what the real pipeline (generate.py, test_questions.py)
    # uses by default.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = get_collection()
    bm25_index = build_bm25_index(collection)

    question = "What house does Harry get sorted into?"
    results = hybrid_search(question, model, collection, bm25_index, n_results=3)

    print(f"Question: {question}\n")
    for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        print(f"{cid}  ({meta['chapter']})")
        print(f"  {doc[:200]}...\n")
