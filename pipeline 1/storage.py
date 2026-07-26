import chromadb
from sentence_transformers import SentenceTransformer

from embedding import build_chunks, EMBEDDING_MODEL_NAME

DB_PATH = "chroma_db"
COLLECTION_NAME = "sorcerers_stone_chunks"


if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = build_chunks(model.tokenizer)
    print(f"Built {len(chunks)} chunks.\n")

    texts = [c["text"] for c in chunks]

    print("Generating embeddings for all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Each chunk needs a unique id. "chunk_0", "chunk_1", ... is enough
    # since we never look these up by anything other than position/search.
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chapter": c["chapter"]} for c in chunks]

    print("\nConnecting to the vector database...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # match the embedding model's training
    )

    print(f"Adding {len(chunks)} records to collection '{COLLECTION_NAME}'...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\nDone. Collection now holds {collection.count()} records.")

    print("\nSample record back from the database (chunk_5):")
    result = collection.get(ids=["chunk_5"])
    print(f"  Chapter: {result['metadatas'][0]['chapter']}")
    print(f"  Text: {result['documents'][0][:200]}...")
