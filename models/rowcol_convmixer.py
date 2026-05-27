import torch
import torch.nn as nn

from models.convmixer import Residual
from models.rowcol import RowColBranch


class RowColOnlyBlock(nn.Module):
    """
    M1 block: RowCol-only spatial mixing.

    This block replaces the original ConvMixer depthwise convolution
    with Row/Column-wise spatial mixing.

    Purpose:
        Ablation study to test whether Row/Col branch alone can replace
        local depthwise convolution.
    """
    def __init__(
        self,
        dim: int,
        feature_size: int,
        rowcol_combine: str = "learnable",
    ):
        super().__init__()

        self.spatial_mixing = Residual(
            RowColBranch(
                dim=dim,
                feature_size=feature_size,
                combine=rowcol_combine,
                use_bn=True,
                use_activation=True,
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


class AdditiveRowColBlock(nn.Module):
    """
    M2 block: ConvMixer local depthwise branch + RowCol branch.

    Structure:
        x = x + F_local(x) + F_rowcol(x)
        x = PointwiseConv(x)

    Interpretation:
        - F_local captures local 2D patterns.
        - F_rowcol captures directional global context.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        feature_size: int,
        rowcol_combine: str = "learnable",
    ):
        super().__init__()

        self.local_branch = nn.Sequential(
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

        self.rowcol_branch = RowColBranch(
            dim=dim,
            feature_size=feature_size,
            combine=rowcol_combine,
            use_bn=True,
            use_activation=True,
        )

        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local_branch(x) + self.rowcol_branch(x)
        x = self.channel_mixing(x)
        return x


class RowColOnlyConvMixer(nn.Module):
    """
    M1: RowCol-only ConvMixer.

    Ablation model:
        X' = X + F_rowcol(X)
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        patch_size: int = 2,
        image_size: int = 32,
        n_classes: int = 10,
        in_channels: int = 3,
        rowcol_combine: str = "learnable",
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size. "
                f"Got image_size={image_size}, patch_size={patch_size}"
            )

        feature_size = image_size // patch_size

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
                RowColOnlyBlock(
                    dim=dim,
                    feature_size=feature_size,
                    rowcol_combine=rowcol_combine,
                )
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


class AdditiveRowColConvMixer(nn.Module):
    """
    M2: Additive RowCol ConvMixer.

    Primary test model:
        X' = X + F_local(X) + F_rowcol(X)

    followed by pointwise channel mixing.
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 5,
        patch_size: int = 2,
        image_size: int = 32,
        n_classes: int = 10,
        in_channels: int = 3,
        rowcol_combine: str = "learnable",
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size. "
                f"Got image_size={image_size}, patch_size={patch_size}"
            )

        feature_size = image_size // patch_size

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
                AdditiveRowColBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    feature_size=feature_size,
                    rowcol_combine=rowcol_combine,
                )
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
