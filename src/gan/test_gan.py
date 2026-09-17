import torch

from generator import Generator
from discriminator import Discriminator


def main():

    print("=" * 60)
    print("GAN ARCHITECTURE TEST")
    print("=" * 60)

    # Create models
    generator = Generator()
    discriminator = Discriminator()

    # Generate random noise
    noise = torch.randn(
        4,
        100,
        1,
        1
    )

    # Generate fake images
    fake_images = generator(noise)

    # Ask discriminator to classify them
    predictions = discriminator(fake_images)

    print("Noise shape       :", noise.shape)
    print("Generated images  :", fake_images.shape)
    print("Discriminator out :", predictions.shape)

    print("=" * 60)
    print("GAN ARCHITECTURE READY")
    print("=" * 60)


if __name__ == "__main__":
    main()