import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    Residual wrapper:
        y = f(x) + x
    """
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    """
    Standard ConvMixer block.

    Structure:
        1) Depthwise spatial mixing with residual connection
        2) Pointwise channel mixing

    Input:
        x: [B, C, H, W]

    Output:
        y: [B, C, H, W]
    """
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()

        self.spatial_mixing = Residual(
            nn.Sequential(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    groups=dim,
                    padding="same",
                    bias=False,
                ),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
        )

        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_mixing(x)
        x = self.channel_mixing(x)
        return x


class ConvMixer(nn.Module):
    """
    M0: ConvMixer baseline.

    This model is the reference baseline for later Row/Col-augmented models.

    Architecture:
        Patch embedding
        -> ConvMixerBlock x depth
        -> Global average pooling
        -> Linear classifier
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 5,
        patch_size: int = 2,
        n_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        self.blocks = nn.Sequential(
            *[
                ConvMixerBlock(dim=dim, kernel_size=kernel_size)
                for _ in range(depth)
            ]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x
