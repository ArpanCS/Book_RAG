from sentence_transformers import SentenceTransformer

from chunking import load_book, split_into_chapters, split_into_paragraphs, chunk_chapter, BOOK_PATH

EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


def build_chunks(tokenizer):
    text = load_book(BOOK_PATH)
    chapters = split_into_chapters(text)

    all_chunks = []
    for heading, body in chapters:
        paragraphs = split_into_paragraphs(body)
        chapter_chunks = chunk_chapter(paragraphs[1:], heading, tokenizer)
        all_chunks.extend(chapter_chunks)
    return all_chunks


if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = build_chunks(model.tokenizer)
    print(f"Built {len(chunks)} chunks.\n")

    texts = [c["text"] for c in chunks]

    print("Generating embeddings for all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"\nShape of the embeddings: {embeddings.shape}")
    print(f"  -> {embeddings.shape[0]} vectors (one per chunk)")
    print(f"  -> {embeddings.shape[1]} numbers per vector (the embedding dimension)\n")

    print("First 10 numbers of chunk #5's vector:")
    print(embeddings[5][:10])
