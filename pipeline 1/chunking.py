import re

BOOK_PATH = "J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt"


def load_book(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chapters(text):
    """
    Splits the whole book into a list of (chapter_name, chapter_text) pairs.
    Chapter headings look like "CHAPTER ONE" on their own line.
    """
    # This pattern matches a line that is just "CHAPTER" followed by a word
    # (ONE, TWO, THREE...), so we can find where each chapter starts.
    pattern = re.compile(r"^CHAPTER [A-Z]+$", re.MULTILINE)

    # findall gives us the matched heading text (e.g. "CHAPTER ONE")
    headings = pattern.findall(text)

    # split() cuts the text at every place the pattern matches, and throws
    # the matched heading away. The very first piece (before "CHAPTER ONE")
    # is just the book title, so we drop it with [1:].
    pieces = pattern.split(text)[1:]

    chapters = []
    for heading, body in zip(headings, pieces):
        chapters.append((heading.strip(), body.strip()))
    return chapters


def split_into_paragraphs(chapter_text):
    """
    Splits one chapter's text into a list of paragraphs.
    Paragraphs are separated by a blank line. Within a paragraph, the
    original file has single line breaks we don't want, so we join those
    back into one continuous line of text.
    """
    raw_paragraphs = chapter_text.split("\n\n")

    paragraphs = []
    for raw in raw_paragraphs:
        raw = raw.strip()
        if not raw:
            continue
        # Replace the internal single line breaks with a space.
        joined = " ".join(raw.split("\n"))
        paragraphs.append(joined)
    return paragraphs


def chunk_chapter(paragraphs, chapter_name, tokenizer, max_tokens=400, overlap_words=40):
    """
    Groups a chapter's paragraphs into chunks of up to max_tokens *model
    tokens* each, without ever splitting a paragraph apart. Each new chunk
    (after the first) starts by repeating the last overlap_words words of
    the previous chunk, so context isn't lost at the seam.

    Size is measured in tokens (via the embedding model's own tokenizer),
    not words, because the embedding model has a hard limit on how many
    tokens it can read at once (max_tokens=400 leaves a safety buffer
    below that limit). Token counts are added up paragraph-by-paragraph
    rather than re-measuring the whole chunk each time; this is a tiny
    approximation, but the safety buffer easily absorbs the difference.

    Note: this assumes no single paragraph is longer than max_tokens. We
    checked that's true for this book (longest paragraph is 199 words).
    A different source text might need a sentence-level fallback for
    paragraphs that are too long on their own.
    """
    chunks = []
    current_words = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_tokens = len(tokenizer.encode(paragraph, add_special_tokens=False))

        # Adding this paragraph would push us over the limit, and we
        # already have something saved up: close off the current chunk
        # before adding this paragraph.
        if current_words and current_tokens + paragraph_tokens > max_tokens:
            chunks.append({
                "chapter": chapter_name,
                "text": " ".join(current_words),
            })
            # Start the next chunk with the tail end of the chunk we just
            # closed, so nearby context carries over.
            current_words = current_words[-overlap_words:]
            current_tokens = len(tokenizer.encode(" ".join(current_words), add_special_tokens=False))

        current_words.extend(paragraph_words)
        current_tokens += paragraph_tokens

    # The last chunk won't have been saved by the loop above, since
    # nothing came after it to trigger the "close it off" step.
    if current_words:
        chunks.append({
            "chapter": chapter_name,
            "text": " ".join(current_words),
        })

    return chunks


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # We need the embedding model's tokenizer *before* chunking, since
    # chunk size is now measured in that model's tokens, not words.
    EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = model.tokenizer

    text = load_book(BOOK_PATH)
    chapters = split_into_chapters(text)

    print(f"Found {len(chapters)} chapters.\n")

    all_chunks = []
    for heading, body in chapters:
        paragraphs = split_into_paragraphs(body)
        # Skip paragraphs[0], which is the chapter title, not story text.
        chapter_chunks = chunk_chapter(paragraphs[1:], heading, tokenizer)
        all_chunks.extend(chapter_chunks)

    print(f"Total chunks across the whole book: {len(all_chunks)}\n")

    word_counts = [len(c["text"].split()) for c in all_chunks]
    token_counts = [len(tokenizer.encode(c["text"], add_special_tokens=False)) for c in all_chunks]

    print(f"Average words per chunk: {sum(word_counts) / len(word_counts):.0f}")
    print(f"Smallest chunk: {min(word_counts)} words")
    print(f"Largest chunk: {max(word_counts)} words\n")

    print(f"Average tokens per chunk: {sum(token_counts) / len(token_counts):.0f}")
    print(f"Largest chunk: {max(token_counts)} tokens (model limit is {model.max_seq_length})")
    over_limit = [t for t in token_counts if t > model.max_seq_length]
    print(f"Chunks exceeding the model's token limit: {len(over_limit)} out of {len(all_chunks)}\n")

    print("Sample chunk (#5):")
    print(f"  Chapter: {all_chunks[5]['chapter']}")
    print(f"  Word count: {len(all_chunks[5]['text'].split())}")
    print(f"  Text: {all_chunks[5]['text'][:300]}...")
