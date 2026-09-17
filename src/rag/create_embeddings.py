from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_DIR = (
    PROJECT_ROOT / "knowledge_base" / "vector_store"
)

CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.json"

EMBEDDINGS_FILE = VECTOR_STORE_DIR / "embeddings.npy"

METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    print("=" * 60)
    print("RAG EMBEDDING GENERATION")
    print("=" * 60)

    print("\nLoading chunks from:")
    print(CHUNKS_FILE)

    if not CHUNKS_FILE.exists():

        print("\nERROR: chunks.json not found.")

        print(
            "\nPlease run:"
        )

        print(
            "python src\\rag\\document_loader.py"
        )

        exit()

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    print(f"\nChunks loaded: {len(chunks)}")

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    print("\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("\nEmbedding generation complete.")

    print(
        f"Embedding matrix shape: {embeddings.shape}"
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(chunks, embeddings):

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    metadata = []

    for chunk in chunks:

        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"]
            }
        )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EMBEDDINGS SAVED")
    print("=" * 60)

    print("\nEmbeddings:")
    print(EMBEDDINGS_FILE)

    print("\nMetadata:")
    print(METADATA_FILE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    chunks = load_chunks()

    embeddings = create_embeddings(
        chunks
    )

    save_embeddings(
        chunks,
        embeddings
    )

    print("\nRAG embedding stage completed successfully.")