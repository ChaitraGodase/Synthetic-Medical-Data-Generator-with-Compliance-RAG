from pathlib import Path

import torch
from torchvision.utils import save_image

from generator import Generator


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# SETTINGS
# ============================================================

LATENT_DIM = 100

# Generate 100 synthetic X-ray images
NUM_IMAGES = 100

IMAGE_CHANNELS = 1


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL PATH
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "dcgan_epoch_020.pth"
)


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD GENERATOR
# ============================================================

def load_generator():

    print("=" * 60)
    print("LOADING TRAINED GENERATOR")
    print("=" * 60)

    print()

    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    print()

    print("Device:")
    print(DEVICE)


    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        print()

        print("ERROR: Checkpoint not found.")

        print()

        print(
            "Expected checkpoint:"
        )

        print(
            CHECKPOINT_PATH
        )

        return None


    # --------------------------------------------------------
    # Create generator
    # --------------------------------------------------------

    generator = Generator(

        latent_dim=LATENT_DIM,

        image_channels=IMAGE_CHANNELS

    ).to(DEVICE)


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE
    )


    generator.load_state_dict(

        checkpoint[
            "generator_state_dict"
        ]

    )


    generator.eval()


    print()

    print(
        "Generator loaded successfully."
    )

    print(
        "Trained epoch:",
        checkpoint.get(
            "epoch",
            "Unknown"
        )
    )


    return generator


# ============================================================
# GENERATE IMAGES
# ============================================================

def generate_images():

    generator = load_generator()


    # --------------------------------------------------------
    # Stop if generator could not be loaded
    # --------------------------------------------------------

    if generator is None:

        return


    print()

    print("=" * 60)

    print("GENERATING SYNTHETIC X-RAYS")

    print("=" * 60)


    # --------------------------------------------------------
    # Generate random noise
    # --------------------------------------------------------

    noise = torch.randn(

        NUM_IMAGES,

        LATENT_DIM,

        1,

        1,

        device=DEVICE
    )


    # --------------------------------------------------------
    # Generate images
    # --------------------------------------------------------

    with torch.no_grad():

        generated_images = generator(
            noise
        )


    # --------------------------------------------------------
    # Save individual images
    # --------------------------------------------------------

    print()

    print(
        "Saving individual images..."
    )


    for index, image in enumerate(
        generated_images
    ):

        output_path = (

            OUTPUT_DIR
            / f"synthetic_xray_{index + 1:03d}.png"

        )


        save_image(

            image,

            output_path,

            normalize=True
        )


    # ========================================================
    # SAVE 10 × 10 GRID
    # ========================================================

    grid_path = (

        OUTPUT_DIR
        / "synthetic_xray_grid.png"

    )


    save_image(

        generated_images,

        grid_path,

        normalize=True,

        nrow=10

    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()

    print("=" * 60)

    print("GENERATION COMPLETE")

    print("=" * 60)

    print()

    print(
        "Number of images generated:",
        NUM_IMAGES
    )

    print()

    print(
        "Individual images saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Grid saved to:"
    )

    print(
        grid_path
    )

    print()

    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    generate_images()