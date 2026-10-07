import torch
import torch.nn as nn

from models.convmixer import Residual
from models.rowcol import RowColBranch, MultiplicativeRowColBranch


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


class WeightedRowColBlock(nn.Module):
    """
    M3 block: Weighted local depthwise branch + RowCol branch.

    M2:
        x = x + F_local(x) + F_rowcol(x)

    M3:
        x = x + alpha_l F_local(x) + beta_l F_rowcol(x)

    alpha_l and beta_l are learnable block-wise scalar parameters.

    Recommended initialization:
        alpha_l = 1.0
        beta_l  = 0.1

    This starts close to ConvMixer and lets the model learn how much
    RowCol context should be injected into each block.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        feature_size: int,
        rowcol_combine: str = "learnable",
        local_init: float = 1.0,
        rowcol_init: float = 0.1,
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

        self.local_scale = nn.Parameter(torch.tensor(float(local_init)))
        self.rowcol_scale = nn.Parameter(torch.tensor(float(rowcol_init)))

        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x
            + self.local_scale * self.local_branch(x)
            + self.rowcol_scale * self.rowcol_branch(x)
        )
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


class WeightedRowColConvMixer(nn.Module):
    """
    M3: Weighted RowCol ConvMixer.

    This model improves M2 by learning the contribution of the local
    depthwise branch and the RowCol branch at each block.

    Block formulation:
        X_l' = X_l + alpha_l F_local(X_l) + beta_l F_rowcol(X_l)

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
        local_init: float = 1.0,
        rowcol_init: float = 0.1,
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
                WeightedRowColBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    feature_size=feature_size,
                    rowcol_combine=rowcol_combine,
                    local_init=local_init,
                    rowcol_init=rowcol_init,
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

    def get_branch_scales(self):
        """
        Return block-wise learned branch scales for analysis.
        """
        scales = []
        for idx, block in enumerate(self.blocks):
            scales.append(
                {
                    "block": idx,
                    "local_scale": float(block.local_scale.detach().cpu().item()),
                    "rowcol_scale": float(block.rowcol_scale.detach().cpu().item()),
                }
            )
        return scales



class WeightedMultiplicativeRowColBlock(nn.Module):
    """
    M5 block: Weighted local branch + multiplicative RowCol lifting branch.

    M3:
        Z_add = X_H 1_W^T + 1_H X_W

    M5:
        Z_mul = (X_H P_W) ⊙ (P_H X_W)

    Block:
        x = x + alpha_l F_local(x) + beta_l F_mul_rowcol(x)
        x = PointwiseConv(x)
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        feature_size: int,
        local_init: float = 1.0,
        rowcol_init: float = 0.1,
        channel_specific_pos: bool = False,
        mul_pre_norm: bool = True,
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

        self.rowcol_branch = MultiplicativeRowColBranch(
            dim=dim,
            feature_size=feature_size,
            channel_specific_pos=channel_specific_pos,
            pre_norm=mul_pre_norm,
            use_bn=True,
            use_activation=True,
        )

        self.local_scale = nn.Parameter(torch.tensor(float(local_init)))
        self.rowcol_scale = nn.Parameter(torch.tensor(float(rowcol_init)))

        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x
            + self.local_scale * self.local_branch(x)
            + self.rowcol_scale * self.rowcol_branch(x)
        )
        x = self.channel_mixing(x)
        return x


class WeightedMultiplicativeRowColConvMixer(nn.Module):
    """
    M5: Outer-Product / Multiplicative RowCol ConvMixer.

    This model replaces the additive RowCol lifting used in M3 with:

        F_mul_rowcol(X)_c = BN(GELU((X_H,c P_W) ⊙ (P_H X_W,c)))

    Full block:
        X_l' = X_l + alpha_l F_local(X_l) + beta_l F_mul_rowcol(X_l)
        X_{l+1} = F_point(X_l')
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
        local_init: float = 1.0,
        rowcol_init: float = 0.1,
        channel_specific_pos: bool = False,
        mul_pre_norm: bool = True,
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
                WeightedMultiplicativeRowColBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    feature_size=feature_size,
                    local_init=local_init,
                    rowcol_init=rowcol_init,
                    channel_specific_pos=channel_specific_pos,
                    mul_pre_norm=mul_pre_norm,
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

    def get_branch_scales(self):
        scales = []
        for idx, block in enumerate(self.blocks):
            item = {
                "block": idx,
                "local_scale": float(block.local_scale.detach().cpu().item()),
                "rowcol_scale": float(block.rowcol_scale.detach().cpu().item()),
            }
            if hasattr(block.rowcol_branch, "get_positional_prior_stats"):
                item.update(block.rowcol_branch.get_positional_prior_stats())
            scales.append(item)
        return scales
