from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ============================================================
# PROJECT PATH
# ============================================================

# Automatically find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# INPUT AND OUTPUT FOLDERS
# ============================================================

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pneumonia"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "pneumonia"


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 64


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_images():

    print("=" * 60)
    print("MEDICAL X-RAY IMAGE PREPROCESSING")
    print("=" * 60)

    print("Project root:")
    print(PROJECT_ROOT)

    print()

    print("Input folder:")
    print(INPUT_DIR)

    print()

    print("Output folder:")
    print(OUTPUT_DIR)

    print("=" * 60)


    # --------------------------------------------------------
    # Check input folder
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        print()
        print("ERROR: Input folder does not exist.")
        print()

        print("Expected folder:")
        print(INPUT_DIR)

        print()
        print("Please make sure your X-ray images are inside:")
        print("data\\raw\\pneumonia")

        return


    # --------------------------------------------------------
    # Find image files
    # --------------------------------------------------------

    image_files = []

    supported_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    for extension in supported_extensions:

        image_files.extend(
            INPUT_DIR.rglob(extension)
        )


    print()
    print("Images found:", len(image_files))


    # --------------------------------------------------------
    # Check whether images exist
    # --------------------------------------------------------

    if len(image_files) == 0:

        print()
        print("ERROR: No images found.")
        print()

        print("Please check:")
        print("1. Your dataset is downloaded.")
        print("2. Images are inside data\\raw\\pneumonia.")
        print("3. The images have .jpg, .jpeg or .png extensions.")

        return


    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    successful = 0
    failed = 0


    print()
    print("Starting preprocessing...")
    print()


    for index, image_path in enumerate(
        tqdm(image_files)
    ):

        try:

            # Open image
            image = Image.open(
                image_path
            )


            # Convert to grayscale
            image = image.convert("L")


            # Resize image
            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )


            # Create a unique output filename
            output_name = (
                f"xray_{index:05d}.png"
            )


            output_path = (
                OUTPUT_DIR /
                output_name
            )


            # Save processed image
            image.save(
                output_path
            )


            successful += 1


        except Exception as error:

            failed += 1

            print()
            print(
                "Failed:",
                image_path
            )

            print(
                "Reason:",
                error
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Images found      :",
        len(image_files)
    )

    print(
        "Successfully saved:",
        successful
    )

    print(
        "Failed            :",
        failed
    )

    print(
        "Output folder     :",
        OUTPUT_DIR
    )

    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    preprocess_images()