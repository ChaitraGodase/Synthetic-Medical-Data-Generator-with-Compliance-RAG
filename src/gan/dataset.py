from pathlib import Path

from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision import transforms


# ============================================================
# MEDICAL IMAGE DATASET
# ============================================================

class MedicalImageDataset(Dataset):

    def __init__(self, image_dir):

        # Convert path to Path object
        self.image_dir = Path(
            image_dir
        )


        # ----------------------------------------------------
        # Find PNG images
        # ----------------------------------------------------

        self.image_files = sorted(
            self.image_dir.glob("*.png")
        )


        # ----------------------------------------------------
        # Image transformation
        # ----------------------------------------------------

        self.transform = transforms.Compose([

            # Convert PIL image → PyTorch tensor
            transforms.ToTensor(),

            # Convert pixel range:
            # 0 → -1
            # 255 → +1
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )

        ])


    # ========================================================
    # NUMBER OF IMAGES
    # ========================================================

    def __len__(self):

        return len(
            self.image_files
        )


    # ========================================================
    # LOAD ONE IMAGE
    # ========================================================

    def __getitem__(self, index):

        # Get image path
        image_path = (
            self.image_files[index]
        )


        # Open image
        image = Image.open(
            image_path
        )


        # Convert to grayscale
        image = image.convert("L")


        # Apply transformations
        image = self.transform(
            image
        )


        return image


# ============================================================
# DATASET TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Find project root automatically
    # --------------------------------------------------------

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[2]
    )


    # --------------------------------------------------------
    # Correct processed dataset path
    # --------------------------------------------------------

    DATASET_DIR = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "pneumonia"
    )


    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = MedicalImageDataset(
        DATASET_DIR
    )


    # ========================================================
    # DISPLAY TEST RESULTS
    # ========================================================

    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print()

    print(
        "Project root:",
        PROJECT_ROOT
    )

    print()

    print(
        "Dataset folder:",
        DATASET_DIR
    )

    print()

    print(
        "Number of images:",
        len(dataset)
    )


    # --------------------------------------------------------
    # Test first image
    # --------------------------------------------------------

    if len(dataset) > 0:

        image = dataset[0]


        print()

        print(
            "First image:",
            dataset.image_files[0]
        )

        print()

        print(
            "Image tensor shape:",
            image.shape
        )

        print(
            "Minimum value:",
            image.min().item()
        )

        print(
            "Maximum value:",
            image.max().item()
        )

        print()

        print("Dataset is working correctly.")


    else:

        print()

        print(
            "ERROR: No processed images found."
        )

        print()

        print(
            "Expected images inside:"
        )

        print(
            DATASET_DIR
        )


    print()
    print("=" * 60)
    print("DATASET TEST COMPLETE")
    print("=" * 60)