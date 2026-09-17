from pathlib import Path

import numpy as np

from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# SYNTHETIC IMAGE DIRECTORY
# ============================================================

SYNTHETIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of images to evaluate
MAX_IMAGES = 100


# ============================================================
# LOAD IMAGES
# ============================================================

def load_images():

    # --------------------------------------------------------
    # Select only individual generated X-rays
    # --------------------------------------------------------

    image_files = sorted(
        SYNTHETIC_DIR.glob(
            "synthetic_xray_[0-9][0-9][0-9].png"
        )
    )


    # Limit to MAX_IMAGES
    image_files = image_files[:MAX_IMAGES]


    print(
        "Images found:",
        len(image_files)
    )


    if len(image_files) < 2:

        print()

        print(
            "ERROR: At least 2 synthetic images "
            "are required."
        )

        return None


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    images = []


    for image_path in image_files:

        try:

            image = Image.open(
                image_path
            ).convert("L")


            image_array = np.array(
                image,
                dtype=np.float32
            )


            # Normalize pixels between 0 and 1
            image_array = (
                image_array / 255.0
            )


            # Flatten 64 × 64 → 4096
            image_array = (
                image_array.flatten()
            )


            images.append(
                image_array
            )


        except Exception as error:

            print(
                "Could not read:",
                image_path
            )

            print(
                "Reason:",
                error
            )


    if len(images) < 2:

        print(
            "ERROR: Not enough valid images."
        )

        return None


    return np.array(images)


# ============================================================
# CALCULATE DIVERSITY
# ============================================================

def calculate_diversity(images):

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarity_matrix = cosine_similarity(
        images
    )


    # --------------------------------------------------------
    # Extract only unique image pairs
    #
    # We don't want:
    # image 1 vs image 1
    # image 2 vs image 2
    # etc.
    # --------------------------------------------------------

    upper_triangle = np.triu_indices(
        len(images),
        k=1
    )


    pairwise_similarities = (
        similarity_matrix[
            upper_triangle
        ]
    )


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    average_similarity = (
        np.mean(
            pairwise_similarities
        )
    )


    minimum_similarity = (
        np.min(
            pairwise_similarities
        )
    )


    maximum_similarity = (
        np.max(
            pairwise_similarities
        )
    )


    similarity_std = (
        np.std(
            pairwise_similarities
        )
    )


    return (
        average_similarity,
        minimum_similarity,
        maximum_similarity,
        similarity_std
    )


# ============================================================
# INTERPRET RESULT
# ============================================================

def interpret_result(
    average_similarity
):

    print()

    print("=" * 60)

    print(
        "DIVERSITY INTERPRETATION"
    )

    print("=" * 60)

    print()


    print(
        f"Average pairwise similarity: "
        f"{average_similarity:.4f}"
    )

    print()


    # --------------------------------------------------------
    # Important:
    #
    # These are heuristic indicators, NOT clinical
    # validation thresholds.
    # --------------------------------------------------------

    if average_similarity >= 0.95:

        print(
            "WARNING: Very high similarity."
        )

        print(
            "The generator may be producing "
            "highly similar images."
        )

        print(
            "Possible mode-collapse behavior."
        )


    elif average_similarity >= 0.85:

        print(
            "Moderate-to-high similarity."
        )

        print(
            "The images contain similar "
            "overall structures."
        )

        print(
            "Further diversity evaluation "
            "may be useful."
        )


    else:

        print(
            "Lower average similarity."
        )

        print(
            "The generated images show "
            "greater variation."
        )


    print()

    print(
        "Note: Similarity alone does not "
        "establish medical realism."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "SYNTHETIC X-RAY DIVERSITY TEST"
    )

    print("=" * 60)

    print()


    print(
        "Synthetic image directory:"
    )

    print(
        SYNTHETIC_DIR
    )

    print()


    # ========================================================
    # LOAD IMAGES
    # ========================================================

    images = load_images()


    if images is None:

        return


    print()

    print(
        "Image matrix shape:",
        images.shape
    )


    # ========================================================
    # CALCULATE DIVERSITY
    # ========================================================

    results = calculate_diversity(
        images
    )


    (
        average_similarity,
        minimum_similarity,
        maximum_similarity,
        similarity_std
    ) = results


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()

    print("=" * 60)

    print(
        "DIVERSITY RESULTS"
    )

    print("=" * 60)

    print()


    print(
        "Number of images:",
        len(images)
    )


    print(
        "Number of image pairs:",
        len(images) * (len(images) - 1) // 2
    )


    print()


    print(
        "Average similarity:",
        f"{average_similarity:.4f}"
    )


    print(
        "Minimum similarity:",
        f"{minimum_similarity:.4f}"
    )


    print(
        "Maximum similarity:",
        f"{maximum_similarity:.4f}"
    )


    print(
        "Similarity standard deviation:",
        f"{similarity_std:.4f}"
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    interpret_result(
        average_similarity
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "DIVERSITY TEST COMPLETE"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()