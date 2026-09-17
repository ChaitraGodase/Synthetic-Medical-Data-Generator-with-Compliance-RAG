from pathlib import Path
import json

import numpy as np
import faiss


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_DIR = (
    PROJECT_ROOT / "knowledge_base" / "vector_store"
)

EMBEDDINGS_FILE = VECTOR_STORE_DIR / "embeddings.npy"

METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"

FAISS_INDEX_FILE = VECTOR_STORE_DIR / "faiss.index"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 60)
print("FAISS VECTOR DATABASE")
print("=" * 60)

print("\nLoading embeddings:")
print(EMBEDDINGS_FILE)

embeddings = np.load(EMBEDDINGS_FILE)

print(f"\nEmbedding shape: {embeddings.shape}")


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata:")
print(METADATA_FILE)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

print(f"Metadata records: {len(metadata)}")


# ============================================================
# PREPARE EMBEDDINGS
# ============================================================

embeddings = embeddings.astype(
    "float32"
)

print("\nEmbedding datatype:")
print(embeddings.dtype)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

embedding_dimension = embeddings.shape[1]

print("\nEmbedding dimension:")
print(embedding_dimension)

print("\nCreating FAISS index...")

index = faiss.IndexFlatIP(
    embedding_dimension
)


# ============================================================
# ADD EMBEDDINGS
# ============================================================

index.add(embeddings)

print("\nVectors added to FAISS:")
print(index.ntotal)


# ============================================================
# SAVE INDEX
# ============================================================

faiss.write_index(
    index,
    str(FAISS_INDEX_FILE)
)

print("\n" + "=" * 60)
print("FAISS INDEX CREATED")
print("=" * 60)

print("\nIndex saved to:")
print(FAISS_INDEX_FILE)

print(f"\nTotal vectors: {index.ntotal}")

print(f"Vector dimension: {embedding_dimension}")

print("\nFAISS database creation completed successfully.")