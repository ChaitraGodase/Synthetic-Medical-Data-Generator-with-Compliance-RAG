
from pathlib import Path
from datetime import datetime, timezone
import json
import random
import re

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_DIR = PROJECT_ROOT / "knowledge_base" / "vector_store"
INDEX_FILE = VECTOR_STORE_DIR / "faiss.index"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "records"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


# ============================================================
# BASIC COMPLIANCE PATTERNS
# ============================================================

IDENTIFIER_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)"
    ),
    "possible_ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
}


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

def load_rag():
    """Load FAISS, metadata, and the sentence embedding model."""

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

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and metadata counts do not match."
        )

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


# ============================================================
# RETRIEVE CLINICAL CONTEXT
# ============================================================

def retrieve_context(query, index, metadata, model):
    """Return the top matching clinical text chunks."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    k = min(TOP_K, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        item = metadata[int(idx)]

        if isinstance(item, dict):
            content = item.get(
                "text",
                item.get("content", "")
            )
            source = item.get(
                "source",
                "Knowledge base"
            )
        else:
            content = str(item)
            source = "Knowledge base"

        results.append({
            "similarity_score": round(float(score), 4),
            "source": source,
            "text": content
        })

    return results


# ============================================================
# GENERATE A FICTIONAL RECORD
# ============================================================

def generate_record(retrieved_context):
    """
    Generate a template-based fictional record.

    This function does not diagnose a patient or determine
    whether the fictional symptoms represent a disease.
    """

    rng = random.SystemRandom()

    symptom_scenarios = [
        ["Cough", "Fatigue"],
        ["Cough", "Fever", "Fatigue"],
        ["Cough", "Shortness of breath"],
        ["Cough", "Chest discomfort during coughing"],
        ["Fever", "Chills", "Cough"],
    ]

    record = {
        "record_type": "SYNTHETIC - FICTIONAL",

        "synthetic_id": (
            f"SYN-{rng.randrange(100000, 1000000)}"
        ),

        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "patient_profile": {
            "age_years": rng.choice(
                [24, 31, 42, 56, 67, 73]
            ),
            "sex": rng.choice(["Female", "Male"]),
            "identity_status": (
                "Fictional; not linked to a real person"
            ),
        },

        "scenario": {
            "topic": "Pneumonia-related educational scenario",
            "symptoms": rng.choice(symptom_scenarios),
            "duration": rng.choice(
                ["2 days", "4 days", "1 week"]
            ),
            "medical_history": (
                "Not specified in this illustrative scenario"
            ),
            "vital_signs": (
                "Not generated; no clinical measurements supplied"
            ),
            "diagnosis": (
                "Not assigned; this record is not a clinical assessment"
            ),
        },

        "rag_context": [
            {
                "source": item["source"],
                "similarity_score": item["similarity_score"],
                "retrieved_text": item["text"],
            }
            for item in retrieved_context
        ],

        "safety_notice": (
            "Entire record is fictional and intended for software "
            "development and educational testing only. It is not a "
            "real medical record, clinical diagnosis, or medical advice. "
            "Synthetic status does not by itself establish privacy "
            "compliance or clinical validity."
        ),
    }

    return record


# ============================================================
# AUTOMATIC COMPLIANCE VALIDATION
# ============================================================

def inspect_value(value, path="$"):
    """Scan record text for potential identifier patterns."""

    findings = []

    if isinstance(value, dict):

        for key, item in value.items():
            findings.extend(
                inspect_value(item, f"{path}.{key}")
            )

    elif isinstance(value, list):

        for index, item in enumerate(value):
            findings.extend(
                inspect_value(item, f"{path}[{index}]")
            )

    elif isinstance(value, str):

        # Avoid false positives from the generated timestamp.
        if path.endswith(".created_at_utc"):
            return findings

        for label, pattern in IDENTIFIER_PATTERNS.items():

            if pattern.search(value):
                findings.append({
                    "type": label,
                    "location": path,
                    "message": (
                        f"Potential {label} pattern detected; "
                        "manual review is required."
                    ),
                })

    return findings


def validate_record(record):
    """Validate required fields, labels, and identifier patterns."""

    errors = []
    warnings = []

    required_fields = [
        "record_type",
        "synthetic_id",
        "created_at_utc",
        "patient_profile",
        "scenario",
        "rag_context",
        "safety_notice",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    if errors:
        return {
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
            "identifier_pattern_findings": 0,
            "disclaimer": (
                "Basic checks do not establish privacy compliance "
                "or medical validity."
            ),
        }

    if record["record_type"] != "SYNTHETIC - FICTIONAL":
        errors.append("Required synthetic label is missing.")

    if not str(record["synthetic_id"]).startswith("SYN-"):
        errors.append("Synthetic ID must begin with SYN-.")

    if not isinstance(record["patient_profile"], dict):
        errors.append("patient_profile must be an object.")

    if not isinstance(record["scenario"], dict):
        errors.append("scenario must be an object.")

    if not isinstance(record["rag_context"], list):
        errors.append("rag_context must be a list.")
    elif not record["rag_context"]:
        warnings.append("No retrieved clinical context was included.")

    safety_notice = str(record["safety_notice"]).lower()

    if "fictional" not in safety_notice:
        errors.append("Safety notice must identify the record as fictional.")

    if "diagnosis" not in safety_notice:
        warnings.append(
            "Safety notice should clarify that this is not a diagnosis."
        )

    findings = inspect_value(record)

    for finding in findings:
        warnings.append(
            f"{finding['type']} pattern at "
            f"{finding['location']}: {finding['message']}"
        )

    if errors:
        status = "FAIL"
    elif warnings:
        status = "REVIEW REQUIRED"
    else:
        status = "PASS"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "identifier_pattern_findings": len(findings),
        "disclaimer": (
            "Passing these basic checks does not prove that a record "
            "is anonymous, privacy-compliant, medically valid, or safe "
            "for clinical use. Human review is still required."
        ),
    }


# ============================================================
# SAVE RECORD AND VALIDATION REPORT
# ============================================================

def save_record_and_report(record, validation):
    """Save the generated record and its validation report."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record_path = OUTPUT_DIR / (
        f"{record['synthetic_id']}.json"
    )

    report_path = OUTPUT_DIR / (
        f"{record['synthetic_id']}_validation.json"
    )

    with open(record_path, "w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            validation,
            file,
            indent=4,
            ensure_ascii=False
        )

    return record_path, report_path


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("SYNTHETIC MEDICAL DATA GENERATOR")
    print("=" * 60)

    print("\n1. Loading RAG components...")

    index, metadata, model = load_rag()

    query = (
        "Common symptoms of pneumonia and clinical evaluation "
        "of pneumonia"
    )

    print("\n2. Retrieving clinical reference information...")

    context = retrieve_context(
        query,
        index,
        metadata,
        model
    )

    if not context:
        raise RuntimeError(
            "No context retrieved; record generation stopped."
        )

    print(f"Retrieved chunks: {len(context)}")

    print("\n3. Generating fictional patient record...")

    record = generate_record(context)

    print("\n4. Running automatic compliance checks...")

    validation = validate_record(record)

    record_path, report_path = save_record_and_report(
        record,
        validation
    )

    print("\n" + "=" * 60)
    print("GENERATION AND VALIDATION RESULTS")
    print("=" * 60)

    print(f"Record ID: {record['synthetic_id']}")
    print(f"Validation status: {validation['status']}")

    print("\nErrors:")
    if validation["errors"]:
        for error in validation["errors"]:
            print(f"- {error}")
    else:
        print("- None")

    print("\nWarnings:")
    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"- {warning}")
    else:
        print("- None")

    print(f"\nRecord saved to:\n{record_path}")
    print(f"\nValidation report saved to:\n{report_path}")

    print("\nIMPORTANT:")
    print(validation["disclaimer"])

    print("\nSynthetic record generation finished.")


if __name__ == "__main__":
    main()
