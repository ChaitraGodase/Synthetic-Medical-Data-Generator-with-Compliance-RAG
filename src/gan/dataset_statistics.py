from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_DIR = Path("data/processed/pneumonia")


def calculate_statistics():

    image_files = list(IMAGE_DIR.glob("*.png"))

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    if len(image_files) == 0:
        print("ERROR: No images found.")
        return

    means = []
    standard_deviations = []

    widths = []
    heights = []

    for image_path in image_files:

        with Image.open(image_path) as image:

            array = np.array(image).astype(np.float32)

            means.append(array.mean())
            standard_deviations.append(array.std())

            width, height = image.size

            widths.append(width)
            heights.append(height)

    print("Number of images :", len(image_files))
    print("Image width      :", widths[0])
    print("Image height     :", heights[0])

    print(
        "Average intensity:",
        round(float(np.mean(means)), 2)
    )

    print(
        "Average std      :",
        round(float(np.mean(standard_deviations)), 2)
    )

    print(
        "Minimum intensity:",
        round(float(min(means)), 2)
    )

    print(
        "Maximum intensity:",
        round(float(max(means)), 2)
    )

    print("=" * 60)


if __name__ == "__main__":
    calculate_statistics()