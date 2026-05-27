import torch
import torch.nn as nn


class RowColBranch(nn.Module):
    """
    Row/Column-wise depthwise convolution branch.

    Input:
        x: [B, C, H, W]

    For CIFAR-10 with image_size=32 and patch_size=2:
        H = W = 16 after patch embedding.

    This branch computes:
        horizontal context: Conv2d(kernel_size=(1, W)) -> [B, C, H, 1]
        vertical context:   Conv2d(kernel_size=(H, 1)) -> [B, C, 1, W]

    The two outputs are broadcast to [B, C, H, W] and combined.
    """
    def __init__(
        self,
        dim: int,
        feature_size: int,
        combine: str = "learnable",
        use_bn: bool = True,
        use_activation: bool = True,
    ):
        super().__init__()

        if combine not in ["sum", "learnable", "softmax"]:
            raise ValueError(f"Unsupported combine mode: {combine}")

        self.dim = dim
        self.feature_size = feature_size
        self.combine = combine

        self.horizontal_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, feature_size),
            groups=dim,
            bias=False,
        )

        self.vertical_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=(feature_size, 1),
            groups=dim,
            bias=False,
        )

        if combine in ["learnable", "softmax"]:
            self.branch_weight = nn.Parameter(torch.tensor([1.0, 1.0]))

        self.activation = nn.GELU() if use_activation else nn.Identity()
        self.bn = nn.BatchNorm2d(dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.horizontal_conv(x)  # [B, C, H, 1]
        v = self.vertical_conv(x)    # [B, C, 1, W]

        if self.combine == "sum":
            y = h + v
        elif self.combine == "learnable":
            y = self.branch_weight[0] * h + self.branch_weight[1] * v
        elif self.combine == "softmax":
            w = torch.softmax(self.branch_weight, dim=0)
            y = w[0] * h + w[1] * v
        else:
            raise RuntimeError("Invalid combine mode.")

        y = self.activation(y)
        y = self.bn(y)
        return y
