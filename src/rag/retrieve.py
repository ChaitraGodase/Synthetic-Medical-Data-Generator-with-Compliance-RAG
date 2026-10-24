
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "knowledge_base" / "vector_store"

INDEX_FILE = VECTOR_STORE_DIR / "faiss.index"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"

# Embedding model used when creating the index
MODEL_NAME = "all-MiniLM-L6-v2"


def load_retriever():
    """Load the FAISS index, metadata, and embedding model."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Run: python src\\rag\\build_faiss.py"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = SentenceTransformer(MODEL_NAME)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but metadata contains {len(metadata)} records."
        )

    return index, metadata, model


def search_documents(query, index, metadata, model, top_k=3):
    """Retrieve the most relevant document chunks for a query."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "metadata": metadata[int(idx)]
        })

    return results


def main():
    print("=" * 60)
    print("RAG DOCUMENT RETRIEVAL TEST")
    print("=" * 60)

    index, metadata, model = load_retriever()

    print(f"\nIndexed chunks: {index.ntotal}")
    print("\nType a question about pneumonia.")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() == "exit":
            print("Retrieval test finished.")
            break

        if not query:
            print("Please enter a question.")
            continue

        results = search_documents(
            query, index, metadata, model, top_k=3
        )

        print("\nTOP MATCHES")
        print("-" * 60)

        for rank, result in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print(f"Similarity score: {result['score']:.4f}")

            record = result["metadata"]

            if isinstance(record, dict):
                print(f"Source: {record.get('source', 'Knowledge base')}")
                content = record.get(
                    "text",
                    record.get("content", str(record))
                )
            else:
                content = str(record)

            print(f"Content:\n{content}")


if __name__ == "__main__":
    main()
