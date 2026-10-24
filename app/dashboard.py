
from pathlib import Path
import json
import re
import subprocess
import sys

import streamlit as st
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_IMAGES = (
    PROJECT_ROOT / "data" / "synthetic" / "generated"
)

SAMPLE_IMAGES = (
    PROJECT_ROOT / "data" / "synthetic" / "samples"
)

RECORD_DIR = (
    PROJECT_ROOT / "data" / "synthetic" / "records"
)

RAG_DIR = (
    PROJECT_ROOT / "knowledge_base" / "vector_store"
)

GENERATOR_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "generation"
    / "generate_synthetic_record.py"
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Synthetic Medical Data Generator",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Synthetic Medical Data Generator")
st.caption(
    "Experimental developer sandbox | DCGAN + RAG + "
    "Synthetic Records + Basic Validation"
)

st.warning(
    "Research and software testing only. Generated images and "
    "records are fictional or experimental and are not for "
    "clinical diagnosis or treatment. Automated checks do not "
    "certify HIPAA compliance or guarantee anonymization."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_original_records():
    """Return original record files, excluding validation reports."""

    if not RECORD_DIR.exists():
        return []

    return sorted(
        [
            path
            for path in RECORD_DIR.glob("SYN-*.json")
            if not path.stem.endswith("_validation")
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_validation_reports():
    """Return validation report files."""

    if not RECORD_DIR.exists():
        return []

    return sorted(
        RECORD_DIR.glob("SYN-*_validation.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_generated_images():
    """Return individual generated X-ray images."""

    if not GENERATED_IMAGES.exists():
        return []

    return sorted(
        [
            path
            for path in GENERATED_IMAGES.glob("synthetic_xray_*.png")
            if re.fullmatch(
                r"synthetic_xray_\d{3}\.png",
                path.name,
            )
        ]
    )


def get_sample_images():
    """Return training sample grids."""

    if not SAMPLE_IMAGES.exists():
        return []

    return sorted(SAMPLE_IMAGES.glob("epoch_*.png"))


def load_json(path):
    """Load a JSON file safely."""

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


# ============================================================
# RAG SEARCH
# ============================================================

@st.cache_resource(show_spinner="Loading clinical search model...")
def load_rag_components():
    """Load the existing FAISS index and embedding model."""

    import faiss
    from sentence_transformers import SentenceTransformer

    index_path = RAG_DIR / "faiss.index"
    metadata_path = RAG_DIR / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run "
            "python src\\rag\\build_faiss.py"
        )

    if not metadata_path.exists():
        raise FileNotFoundError("metadata.json was not found.")

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and metadata counts do not match."
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, metadata, model


def search_knowledge_base(query, top_k=3):
    """Retrieve relevant reference chunks for a query."""

    index, metadata, model = load_rag_components()

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, index.ntotal)

    scores, indices = index.search(embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = metadata[int(idx)]

        if isinstance(item, dict):
            text = item.get(
                "text",
                item.get("content", str(item)),
            )
            source = item.get("source", "Knowledge base")
        else:
            text = str(item)
            source = "Knowledge base"

        results.append({
            "score": float(score),
            "source": source,
            "text": text,
        })

    return results


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "X-ray Gallery",
        "Clinical Knowledge Search",
        "Generate Patient Record",
        "Validation Reports",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Local development environment")


# ============================================================
# PAGE 1: PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.header("Project Overview")

    records = get_original_records()
    reports = get_validation_reports()
    images = get_generated_images()
    samples = get_sample_images()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Generated X-rays", len(images))
    col2.metric("Synthetic Records", len(records))
    col3.metric("Validation Reports", len(reports))
    col4.metric("Training Sample Grids", len(samples))

    st.subheader("System Components")

    st.markdown(
        """
        - **DCGAN:** Generates experimental synthetic X-ray images.
        - **RAG:** Retrieves relevant passages from the local knowledge base.
        - **Record Generator:** Creates template-based fictional patient scenarios.
        - **Validation:** Checks required fields and selected identifier patterns.
        """
    )

    st.info(
        "The record generator currently uses predefined scenarios. "
        "It does not generate clinically validated patient histories."
    )

    st.subheader("Latest Synthetic Records")

    if records:
        for record_path in records[:5]:
            data = load_json(record_path)

            if data:
                with st.expander(record_path.name):
                    st.write(
                        "Synthetic ID:",
                        data.get("synthetic_id", "Unknown"),
                    )
                    st.write(
                        "Created:",
                        data.get("created_at_utc", "Unknown"),
                    )
                    st.json(data.get("scenario", {}))
    else:
        st.write("No synthetic records found yet.")


# ============================================================
# PAGE 2: X-RAY GALLERY
# ============================================================

elif page == "X-ray Gallery":

    st.header("Synthetic X-ray Gallery")

    images = get_generated_images()
    sample_images = get_sample_images()

    if images:

        st.write(f"Individual generated images: {len(images)}")

        page_size = st.selectbox(
            "Images per page",
            [10, 20, 50],
            index=0,
        )

        total_pages = max(
            1,
            (len(images) + page_size - 1) // page_size,
        )

        current_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        )

        start = (current_page - 1) * page_size
        selected_images = images[start:start + page_size]

        columns = st.columns(5)

        for i, image_path in enumerate(selected_images):
            with columns[i % 5]:
                try:
                    image = Image.open(image_path)
                    st.image(
                        image,
                        caption=image_path.name,
                        use_container_width=True,
                    )
                except OSError:
                    st.error(f"Could not open {image_path.name}")

    else:
        st.info(
            "No individual generated images found. "
            "Run python src\\gan\\generate.py first."
        )

    st.subheader("Training Samples")

    if sample_images:
        selected_sample = st.selectbox(
            "Select an epoch sample grid",
            sample_images,
            format_func=lambda path: path.name,
        )

        try:
            st.image(
                Image.open(selected_sample),
                caption=selected_sample.name,
                use_container_width=True,
            )
        except OSError:
            st.error("Could not open the selected sample grid.")

    else:
        st.write("No epoch sample grids found.")


# ============================================================
# PAGE 3: CLINICAL KNOWLEDGE SEARCH
# ============================================================

elif page == "Clinical Knowledge Search":

    st.header("Clinical Knowledge Search")

    st.write(
        "Search your local knowledge base and inspect the "
        "retrieved reference passages."
    )

    query = st.text_input(
        "Enter a question",
        placeholder="What are the common symptoms of pneumonia?",
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("Search Knowledge Base", type="primary"):

        if not query.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Searching clinical references..."):
                    results = search_knowledge_base(
                        query.strip(),
                        top_k=top_k,
                    )

                if not results:
                    st.info("No matching chunks were returned.")

                for rank, result in enumerate(results, start=1):
                    with st.expander(
                        f"Result {rank} — "
                        f"similarity {result['score']:.4f}",
                        expanded=(rank == 1),
                    ):
                        st.caption(
                            f"Source: {result['source']}"
                        )
                        st.write(result["text"])

                st.caption(
                    "Similarity scores rank text matches; they are "
                    "not probabilities or measures of medical correctness."
                )

            except Exception as exc:
                st.error(f"Search failed: {exc}")


# ============================================================
# PAGE 4: GENERATE SYNTHETIC PATIENT RECORD
# ============================================================

elif page == "Generate Patient Record":

    st.header("Generate Synthetic Patient Record")

    st.write(
        "Create a fictional, template-based record and automatically "
        "run the existing basic validation checks."
    )

    st.warning(
        "This action creates a new JSON record and a validation report "
        "in data/synthetic/records."
    )

    if st.button("Generate Record", type="primary"):

        if not GENERATOR_SCRIPT.exists():
            st.error(
                f"Generator script not found: {GENERATOR_SCRIPT}"
            )

        else:
            try:
                with st.spinner(
                    "Retrieving context, generating and validating..."
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(GENERATOR_SCRIPT),
                        ],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=False,
                    )

                if result.stdout:
                    st.code(result.stdout, language="text")

                if result.returncode != 0:
                    st.error("Record generation failed.")

                    if result.stderr:
                        st.code(result.stderr, language="text")

                else:
                    st.success(
                        "Generator finished. Check the validation "
                        "status in the output above."
                    )

                    records = get_original_records()

                    if records:
                        latest_record = records[0]
                        record_data = load_json(latest_record)

                        if record_data:
                            st.subheader("Generated Record")
                            st.json(record_data)

                            st.download_button(
                                "Download Synthetic Record JSON",
                                data=json.dumps(
                                    record_data,
                                    indent=2,
                                    ensure_ascii=False,
                                ),
                                file_name=latest_record.name,
                                mime="application/json",
                            )

                            report_path = latest_record.with_name(
                                latest_record.stem + "_validation.json"
                            )

                            report_data = load_json(report_path)

                            if report_data:
                                st.subheader("Validation Result")
                                st.write(
                                    "Status:",
                                    report_data.get(
                                        "status",
                                        "Unknown",
                                    ),
                                )
                                st.json(report_data)

                            st.info(
                                "A successful basic validation is not "
                                "proof of anonymization, HIPAA compliance, "
                                "or clinical validity."
                            )

            except subprocess.TimeoutExpired:
                st.error(
                    "Generation timed out. Check your terminal "
                    "and try again."
                )

            except Exception as exc:
                st.error(f"Could not generate the record: {exc}")


# ============================================================
# PAGE 5: VALIDATION REPORTS
# ============================================================

elif page == "Validation Reports":

    st.header("Validation Reports")

    reports = get_validation_reports()

    if not reports:
        st.info(
            "No validation reports found. Generate a synthetic "
            "record to create one."
        )

    else:
        report_options = {
            path.name: path for path in reports
        }

        selected_name = st.selectbox(
            "Select a validation report",
            list(report_options.keys()),
        )

        selected_path = report_options[selected_name]
        report = load_json(selected_path)

        if report is None:
            st.error("Could not read this validation report.")

        else:
            status = report.get("status", "Unknown")

            if status == "PASS":
                st.success(f"Validation status: {status}")
            elif status == "REVIEW REQUIRED":
                st.warning(f"Validation status: {status}")
            else:
                st.error(f"Validation status: {status}")

            st.subheader("Validation Details")
            st.json(report)

            st.download_button(
                "Download Validation Report",
                data=json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name=selected_path.name,
                mime="application/json",
            )

    st.caption(
        "These reports contain basic automated checks only. "
        "They are not compliance certifications."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Synthetic Medical Data Generator | Educational and software "
    "development use only | Not for clinical diagnosis."
)
