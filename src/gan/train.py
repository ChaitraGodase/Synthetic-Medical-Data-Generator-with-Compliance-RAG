from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.utils import save_image

from dataset import MedicalImageDataset
from generator import Generator
from discriminator import Discriminator


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "pneumonia"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT /
    "models" /
    "checkpoints"
)

SAMPLE_DIR = (
    PROJECT_ROOT /
    "data" /
    "synthetic" /
    "samples"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 64

IMAGE_SIZE = 64

LATENT_DIM = 100

# Start with 5 epochs for testing
NUM_EPOCHS = 20

LEARNING_RATE = 0.0002

BETA1 = 0.5

BETA2 = 0.999

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(model):

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d
            )
        ):

            nn.init.normal_(
                module.weight.data,
                0.0,
                0.02
            )

        elif isinstance(
            module,
            nn.BatchNorm2d
        ):

            nn.init.normal_(
                module.weight.data,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias.data,
                0
            )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train():

    print("=" * 70)
    print("SYNTHETIC MEDICAL X-RAY GAN")
    print("DCGAN TRAINING")
    print("=" * 70)

    print()

    print("Project root:")
    print(PROJECT_ROOT)

    print()

    print("Dataset folder:")
    print(IMAGE_DIR)

    print()

    print("Device:", DEVICE)


    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "Using CPU"
        )


    # ========================================================
    # LOAD DATASET
    # ========================================================

    dataset = MedicalImageDataset(
        IMAGE_DIR
    )


    print()

    print(
        "Dataset size:",
        len(dataset)
    )


    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if len(dataset) == 0:

        print()

        print(
            "ERROR: No processed images found."
        )

        print()

        print(
            "Expected folder:"
        )

        print(
            IMAGE_DIR
        )

        return


    # ========================================================
    # DATA LOADER
    # ========================================================

    dataloader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        drop_last=True
    )


    print(
        "Number of batches:",
        len(dataloader)
    )


    # ========================================================
    # CREATE GENERATOR
    # ========================================================

    generator = Generator(

        latent_dim=LATENT_DIM,

        image_channels=1

    ).to(DEVICE)


    # ========================================================
    # CREATE DISCRIMINATOR
    # ========================================================

    discriminator = Discriminator(

        image_channels=1

    ).to(DEVICE)


    # ========================================================
    # INITIALIZE WEIGHTS
    # ========================================================

    initialize_weights(
        generator
    )

    initialize_weights(
        discriminator
    )


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.BCELoss()


    # ========================================================
    # OPTIMIZERS
    # ========================================================

    optimizer_generator = torch.optim.Adam(

        generator.parameters(),

        lr=LEARNING_RATE,

        betas=(
            BETA1,
            BETA2
        )
    )


    optimizer_discriminator = torch.optim.Adam(

        discriminator.parameters(),

        lr=LEARNING_RATE,

        betas=(
            BETA1,
            BETA2
        )
    )


    # ========================================================
    # FIXED NOISE
    # ========================================================

    fixed_noise = torch.randn(

        16,

        LATENT_DIM,

        1,

        1,

        device=DEVICE
    )


    # ========================================================
    # START TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)


    for epoch in range(NUM_EPOCHS):

        generator.train()

        discriminator.train()


        generator_loss_total = 0.0

        discriminator_loss_total = 0.0


        # ====================================================
        # BATCH LOOP
        # ====================================================

        for batch_index, real_images in enumerate(
            dataloader
        ):


            # ------------------------------------------------
            # Move images to CPU/GPU
            # ------------------------------------------------

            real_images = real_images.to(
                DEVICE
            )


            current_batch_size = (
                real_images.size(0)
            )


            # =================================================
            # TRAIN DISCRIMINATOR
            # =================================================

            optimizer_discriminator.zero_grad()


            # Real labels = 1
            real_labels = torch.ones(

                current_batch_size,

                device=DEVICE
            )


            # Fake labels = 0
            fake_labels = torch.zeros(

                current_batch_size,

                device=DEVICE
            )


            # -------------------------------------------------
            # Discriminator on real images
            # -------------------------------------------------

            real_output = discriminator(
                real_images
            )


            real_loss = criterion(

                real_output,

                real_labels
            )


            # -------------------------------------------------
            # Generate fake images
            # -------------------------------------------------

            noise = torch.randn(

                current_batch_size,

                LATENT_DIM,

                1,

                1,

                device=DEVICE
            )


            fake_images = generator(
                noise
            )


            # -------------------------------------------------
            # Discriminator on fake images
            # -------------------------------------------------

            fake_output = discriminator(

                fake_images.detach()
            )


            fake_loss = criterion(

                fake_output,

                fake_labels
            )


            # -------------------------------------------------
            # Total discriminator loss
            # -------------------------------------------------

            discriminator_loss = (

                real_loss +
                fake_loss

            ) / 2


            discriminator_loss.backward()

            optimizer_discriminator.step()


            # =================================================
            # TRAIN GENERATOR
            # =================================================

            optimizer_generator.zero_grad()


            generator_output = discriminator(

                fake_images
            )


            generator_loss = criterion(

                generator_output,

                real_labels
            )


            generator_loss.backward()

            optimizer_generator.step()


            # -------------------------------------------------
            # Store losses
            # -------------------------------------------------

            generator_loss_total += (
                generator_loss.item()
            )

            discriminator_loss_total += (
                discriminator_loss.item()
            )


        # ====================================================
        # AVERAGE LOSSES
        # ====================================================

        average_generator_loss = (

            generator_loss_total /
            len(dataloader)
        )


        average_discriminator_loss = (

            discriminator_loss_total /
            len(dataloader)
        )


        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print(

            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"| D Loss: "
            f"{average_discriminator_loss:.4f} "
            f"| G Loss: "
            f"{average_generator_loss:.4f}"

        )


        # ====================================================
        # GENERATE SAMPLE IMAGES
        # ====================================================

        generator.eval()


        with torch.no_grad():

            generated_images = generator(

                fixed_noise
            )


        sample_path = (

            SAMPLE_DIR /
            f"epoch_{epoch + 1:03d}.png"

        )


        save_image(

            generated_images,

            sample_path,

            normalize=True,

            nrow=4
        )


        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        checkpoint_path = (

            CHECKPOINT_DIR /
            f"dcgan_epoch_{epoch + 1:03d}.pth"

        )


        torch.save(

            {

                "epoch":
                    epoch + 1,

                "generator_state_dict":
                    generator.state_dict(),

                "discriminator_state_dict":
                    discriminator.state_dict(),

                "generator_optimizer":
                    optimizer_generator.state_dict(),

                "discriminator_optimizer":
                    optimizer_discriminator.state_dict(),

                "generator_loss":
                    average_generator_loss,

                "discriminator_loss":
                    average_discriminator_loss,

            },

            checkpoint_path
        )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()

    print("=" * 70)

    print("TRAINING COMPLETE")

    print("=" * 70)

    print()

    print(
        "Checkpoints saved to:"
    )

    print(
        CHECKPOINT_DIR
    )

    print()

    print(
        "Generated samples saved to:"
    )

    print(
        SAMPLE_DIR
    )

    print()

    print("=" * 70)


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train()