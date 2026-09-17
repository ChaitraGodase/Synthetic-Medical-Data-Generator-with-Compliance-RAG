import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(
        self,
        image_channels=1
    ):

        super().__init__()

        self.model = nn.Sequential(

            # 64×64 → 32×32
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 32×32 → 16×16
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 16×16 → 8×8
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8×8 → 4×4
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 4×4 → 1×1
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.Sigmoid()
        )

    def forward(self, image):

        return self.model(image).view(-1)


if __name__ == "__main__":

    discriminator = Discriminator()

    images = torch.randn(
        4,
        1,
        64,
        64
    )

    predictions = discriminator(images)

    print("=" * 50)
    print("DISCRIMINATOR TEST")
    print("=" * 50)

    print("Input shape :", images.shape)
    print("Output shape:", predictions.shape)

    print("=" * 50)