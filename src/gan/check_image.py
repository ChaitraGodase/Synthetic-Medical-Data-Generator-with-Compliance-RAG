from pathlib import Path

from PIL import Image


IMAGE_DIR = Path("data/processed/pneumonia")


def check_images():
    image_files = list(IMAGE_DIR.glob("*.png"))

    print("=" * 50)
    print("PROCESSED IMAGE CHECK")
    print("=" * 50)

    print("Images found:", len(image_files))

    if len(image_files) == 0:
        print("ERROR: No processed images found.")
        return

    # Check first 5 images
    for image_path in image_files[:5]:

        with Image.open(image_path) as image:

            print("\nFile:", image_path.name)
            print("Size:", image.size)
            print("Mode:", image.mode)

    print("=" * 50)


if __name__ == "__main__":
    check_images()