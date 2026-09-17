from pathlib import Path
import json
import re


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIR = PROJECT_ROOT / "knowledge_base" / "documents"

OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "vector_store"

OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


# ============================================================
# CHUNK SETTINGS
# ============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Clean unnecessary spaces and blank lines.
    """

    # Replace multiple spaces with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# LOAD DOCUMENT
# ============================================================

def load_documents():

    documents = []

    markdown_files = sorted(DOCUMENT_DIR.glob("*.md"))

    print("=" * 60)
    print("RAG DOCUMENT LOADER")
    print("=" * 60)

    print(f"\nDocument directory:")
    print(DOCUMENT_DIR)

    print(f"\nDocuments found: {len(markdown_files)}")

    if len(markdown_files) == 0:
        print("\nNo Markdown documents found.")
        return documents

    for file_path in markdown_files:

        print(f"\nLoading: {file_path.name}")

        text = file_path.read_text(
            encoding="utf-8"
        )

        text = clean_text(text)

        documents.append(
            {
                "source": file_path.name,
                "text": text
            }
        )

        print(f"Characters: {len(text)}")

    return documents


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def create_chunks(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE ALL CHUNKS
# ============================================================

def build_chunk_database(documents):

    all_chunks = []

    chunk_id = 0

    for document in documents:

        source = document["source"]

        text = document["text"]

        chunks = create_chunks(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        print(f"\nSource: {source}")
        print(f"Chunks created: {len(chunks)}")

        for chunk in chunks:

            chunk_id += 1

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": chunk
                }
            )

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"\nTotal chunks: {len(chunks)}")

    print("\nSaved to:")

    print(OUTPUT_FILE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    documents = load_documents()

    if len(documents) == 0:
        print("\nNo documents available.")
        exit()

    chunks = build_chunk_database(
        documents
    )

    save_chunks(chunks)

    print("\nRAG document processing finished successfully.")