from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_DIR = Path("data/processed/pneumonia")


def show_images():
    image_files = list(IMAGE_DIR.glob("*.png"))

    if len(image_files) == 0:
        print("ERROR: No processed images found.")
        return

    # Select up to 9 images
    selected_images = image_files[:9]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9, 9)
    )

    for ax, image_path in zip(axes.flat, selected_images):

        image = Image.open(image_path)

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.set_title(image_path.name)

        ax.axis("off")

    # Hide unused axes
    for ax in axes.flat[len(selected_images):]:
        ax.axis("off")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    show_images()