import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(
        self,
        latent_dim=100,
        image_channels=1
    ):

        super().__init__()

        self.model = nn.Sequential(

            # 100 → 4×4
            nn.ConvTranspose2d(
                latent_dim,
                512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(
                64,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Tanh()
        )

    def forward(self, z):

        return self.model(z)


if __name__ == "__main__":

    generator = Generator()

    noise = torch.randn(
        4,
        100,
        1,
        1
    )

    fake_images = generator(noise)

    print("=" * 50)
    print("GENERATOR TEST")
    print("=" * 50)

    print("Input shape :", noise.shape)
    print("Output shape:", fake_images.shape)

    print("=" * 50)