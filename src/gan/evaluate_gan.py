from pathlib import Path

import numpy as np

from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATASET PATHS
# ============================================================

REAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pneumonia"
)

SYNTHETIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
)


# ============================================================
# IMAGE STATISTICS FUNCTION
# ============================================================

def calculate_statistics(image_dir, pattern="*.png"):

    image_files = sorted(
        image_dir.glob(pattern)
    )

    if len(image_files) == 0:

        return None


    means = []
    stds = []
    sizes = []


    for image_path in image_files:

        try:

            # ------------------------------------------------
            # Open image
            # ------------------------------------------------

            image = Image.open(
                image_path
            ).convert("L")


            # ------------------------------------------------
            # Convert to NumPy
            # ------------------------------------------------

            image_array = np.array(
                image,
                dtype=np.float32
            )


            # ------------------------------------------------
            # Calculate statistics
            # ------------------------------------------------

            means.append(
                image_array.mean()
            )

            stds.append(
                image_array.std()
            )

            sizes.append(
                image.size
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


    return {

        "count": len(means),

        "mean_intensity": np.mean(
            means
        ),

        "std_intensity": np.mean(
            stds
        ),

        "sizes": sizes

    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    name,
    statistics
):

    print()

    print("-" * 60)

    print(name)

    print("-" * 60)


    if statistics is None:

        print(
            "No images found."
        )

        return


    print(
        "Number of images:",
        statistics["count"]
    )


    print(
        "Average intensity:",
        f"{statistics['mean_intensity']:.2f}"
    )


    print(
        "Average intensity std:",
        f"{statistics['std_intensity']:.2f}"
    )


    unique_sizes = set(
        statistics["sizes"]
    )


    print(
        "Image sizes:",
        unique_sizes
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print("GAN DATASET EVALUATION")

    print("=" * 60)


    print()

    print(
        "Real dataset:",
        REAL_DIR
    )


    print()

    print(
        "Synthetic dataset:",
        SYNTHETIC_DIR
    )


    # ========================================================
    # REAL IMAGE STATISTICS
    # ========================================================

    real_statistics = calculate_statistics(

        REAL_DIR,

        pattern="*.png"

    )


    # ========================================================
    # SYNTHETIC IMAGE STATISTICS
    # ========================================================

    # IMPORTANT:
    # Only evaluate individual generated X-rays.
    #
    # This excludes:
    # synthetic_xray_grid.png
    #
    # Therefore only:
    # synthetic_xray_001.png
    # synthetic_xray_002.png
    # ...
    # synthetic_xray_100.png
    #
    # are evaluated.

    synthetic_statistics = calculate_statistics(

        SYNTHETIC_DIR,

        pattern="synthetic_xray_[0-9][0-9][0-9].png"

    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print_statistics(

        "REAL X-RAY DATASET",

        real_statistics

    )


    print_statistics(

        "SYNTHETIC X-RAY DATASET",

        synthetic_statistics

    )


    # ========================================================
    # COMPARISON
    # ========================================================

    if (

        real_statistics is not None

        and

        synthetic_statistics is not None

    ):

        print()

        print("=" * 60)

        print(
            "REAL VS SYNTHETIC COMPARISON"
        )

        print("=" * 60)


        # ----------------------------------------------------
        # Mean intensity difference
        # ----------------------------------------------------

        mean_difference = abs(

            real_statistics[
                "mean_intensity"
            ]

            -

            synthetic_statistics[
                "mean_intensity"
            ]

        )


        # ----------------------------------------------------
        # Intensity standard deviation difference
        # ----------------------------------------------------

        std_difference = abs(

            real_statistics[
                "std_intensity"
            ]

            -

            synthetic_statistics[
                "std_intensity"
            ]

        )


        print()

        print(
            "Mean intensity difference:",
            f"{mean_difference:.2f}"
        )


        print(
            "Intensity std difference:",
            f"{std_difference:.2f}"
        )


        print()

        print(
            "Real image count:",
            real_statistics[
                "count"
            ]
        )


        print(
            "Synthetic image count:",
            synthetic_statistics[
                "count"
            ]
        )


        # ----------------------------------------------------
        # Resolution comparison
        # ----------------------------------------------------

        print()

        print(
            "Real image resolution:",
            set(
                real_statistics[
                    "sizes"
                ]
            )
        )


        print(
            "Synthetic image resolution:",
            set(
                synthetic_statistics[
                    "sizes"
                ]
            )
        )


    # ========================================================
    # FINAL NOTE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "EVALUATION NOTE"
    )

    print("=" * 60)

    print()

    print(
        "The synthetic grid image is excluded."
    )

    print(
        "Only individual 64x64 synthetic X-rays"
    )

    print(
        "are included in the statistics."
    )


    print()

    print("=" * 60)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()