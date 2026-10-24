
from pathlib import Path
import json
import re
import sys


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECORD_DIR = PROJECT_ROOT / "data" / "synthetic" / "records"

REQUIRED_FIELDS = [
    "record_type",
    "synthetic_id",
    "created_at_utc",
    "patient_profile",
    "scenario",
    "rag_context",
    "safety_notice",
]


# ============================================================
# POTENTIAL IDENTIFIER PATTERNS
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
# IDENTIFIER SCANNING
# ============================================================

def inspect_value(value, path="$"):
    """Find possible identifier patterns in record values."""

    findings = []

    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            findings.extend(inspect_value(item, child_path))

    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            findings.extend(inspect_value(item, child_path))

    elif isinstance(value, str):

        # Ignore the generated timestamp to avoid false positives.
        if path.endswith(".created_at_utc"):
            return findings

        for label, pattern in IDENTIFIER_PATTERNS.items():
            if pattern.search(value):
                findings.append({
                    "type": label,
                    "location": path,
                    "message": (
                        f"Potential {label} pattern detected. "
                        "Manual review is required."
                    ),
                })

    return findings


# ============================================================
# RECORD VALIDATION
# ============================================================

def validate_record(record):
    """Check required fields, labels, and potential identifiers."""

    errors = []
    warnings = []

    if not isinstance(record, dict):
        return {
            "status": "FAIL",
            "errors": ["Record must be a JSON object."],
            "warnings": [],
            "identifier_pattern_findings": 0,
            "disclaimer": (
                "Basic validation only; this does not establish "
                "privacy compliance or medical validity."
            ),
        }

    # Check required fields.
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    # Stop if the main record structure is incomplete.
    if errors:
        return {
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
            "identifier_pattern_findings": 0,
            "disclaimer": (
                "Basic validation only; this does not establish "
                "privacy compliance or medical validity."
            ),
        }

    # Synthetic label.
    if record["record_type"] != "SYNTHETIC - FICTIONAL":
        errors.append(
            "Record is missing the required synthetic label."
        )

    # Synthetic ID.
    if not str(record["synthetic_id"]).startswith("SYN-"):
        errors.append(
            "Synthetic ID does not use the SYN- prefix."
        )

    # Field types.
    if not isinstance(record["patient_profile"], dict):
        errors.append("patient_profile must be a JSON object.")

    if not isinstance(record["scenario"], dict):
        errors.append("scenario must be a JSON object.")

    if not isinstance(record["rag_context"], list):
        errors.append("rag_context must be a JSON list.")
    elif not record["rag_context"]:
        warnings.append(
            "No retrieved reference chunks were included."
        )

    # Safety notice.
    safety_notice = str(record.get("safety_notice", ""))
    safety_lower = safety_notice.lower()

    if "fictional" not in safety_lower:
        errors.append(
            "Safety notice does not identify the record as fictional."
        )

    if "diagnosis" not in safety_lower:
        warnings.append(
            "Safety notice should clarify that the record is not "
            "a clinical diagnosis."
        )

    # Identifier pattern checks.
    findings = inspect_value(record)

    for finding in findings:
        warnings.append(
            f"{finding['type']} pattern at {finding['location']}: "
            f"{finding['message']}"
        )

    # Final status.
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
            "Passing these checks does not prove that a record is "
            "anonymous, privacy-compliant, medically valid, or safe "
            "for clinical use. Pattern-based checks can miss "
            "identifiers and can produce false positives. "
            "Human review is still required."
        ),
    }


# ============================================================
# SELECT ORIGINAL SYNTHETIC RECORD
# ============================================================

def select_record_path():

    # If a path was supplied explicitly, use that path.
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    # Only select original SYN-*.json records.
    # Exclude validation reports and other derived files.
    files = [
        path
        for path in RECORD_DIR.glob("SYN-*.json")
        if not path.stem.endswith("_validation")
    ]

    if not files:
        raise FileNotFoundError(
            f"No original synthetic records found in {RECORD_DIR}"
        )

    return max(files, key=lambda path: path.stat().st_mtime)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("SYNTHETIC RECORD VALIDATION")
    print("=" * 60)

    try:
        record_path = select_record_path()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    if not record_path.exists():
        print(f"\nERROR: Record file not found: {record_path}")
        sys.exit(1)

    # Never validate a validation report as a patient record.
    if record_path.stem.endswith("_validation"):
        print(
            "\nERROR: You selected a validation report, not an "
            "original synthetic record."
        )
        sys.exit(1)

    # Load the record.
    try:
        with open(record_path, "r", encoding="utf-8") as file:
            record = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"\nERROR: Could not read record: {exc}")
        sys.exit(1)

    # Run checks.
    report = validate_record(record)

    print(f"\nFile: {record_path}")
    print(f"Status: {report['status']}")

    print("\nErrors:")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
    else:
        print("- None")

    print("\nWarnings:")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"- {warning}")
    else:
        print("- None")

    print(
        "\nPotential identifier patterns detected: "
        f"{report['identifier_pattern_findings']}"
    )

    print("\nImportant limitation:")
    print(report["disclaimer"])

    # Save a separate validation report.
    report_path = record_path.with_name(
        record_path.stem + "_validation.json"
    )

    try:
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )
    except OSError as exc:
        print(f"\nERROR: Could not save report: {exc}")
        sys.exit(1)

    print(f"\nValidation report saved to: {report_path}")


if __name__ == "__main__":
    main()
