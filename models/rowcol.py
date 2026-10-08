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
        pre_norm: bool = False,
    ):
        super().__init__()

        if combine not in ["sum", "learnable", "softmax"]:
            raise ValueError(f"Unsupported combine mode: {combine}")

        self.dim = dim
        self.feature_size = feature_size
        self.combine = combine
        self.pre_norm = pre_norm

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

        if pre_norm:
            self.h_norm = nn.BatchNorm2d(dim)
            self.w_norm = nn.BatchNorm2d(dim)

        self.activation = nn.GELU() if use_activation else nn.Identity()
        self.bn = nn.BatchNorm2d(dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.horizontal_conv(x)  # [B, C, H, 1]
        v = self.vertical_conv(x)    # [B, C, 1, W]

        if self.pre_norm:
            h = self.h_norm(h)
            v = self.w_norm(v)

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



class MultiplicativeRowColBranch(nn.Module):
    """
    Multiplicative Row/Column-wise branch.

    Implements the outer-product / multiplicative lifting idea:

        R = X_H P_W
        C = P_H X_W
        Z = R ⊙ C

    where:
        X_H: row-wise projected feature,    [B, C, H, 1]
        X_W: column-wise projected feature, [B, C, 1, W]
        P_H: learnable height prior,        [1, 1 or C, H, 1]
        P_W: learnable width prior,         [1, 1 or C, 1, W]

    Code-level broadcasting:
        h = horizontal_conv(x)  # [B, C, H, 1]
        w = vertical_conv(x)    # [B, C, 1, W]

        row_map = h * p_w       # [B, C, H, W]
        col_map = p_h * w       # [B, C, H, W]
        z = row_map * col_map   # [B, C, H, W]
    """
    def __init__(
        self,
        dim: int,
        feature_size: int,
        channel_specific_pos: bool = False,
        pre_norm: bool = True,
        use_bn: bool = True,
        use_activation: bool = True,
        use_pos_prior: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.feature_size = feature_size
        self.channel_specific_pos = channel_specific_pos
        self.pre_norm = pre_norm
        self.use_pos_prior = use_pos_prior

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

        if use_pos_prior:
            if channel_specific_pos:
                self.p_h = nn.Parameter(torch.ones(1, dim, feature_size, 1))
                self.p_w = nn.Parameter(torch.ones(1, dim, 1, feature_size))
            else:
                self.p_h = nn.Parameter(torch.ones(1, 1, feature_size, 1))
                self.p_w = nn.Parameter(torch.ones(1, 1, 1, feature_size))

        self.h_norm = nn.BatchNorm2d(dim) if pre_norm else nn.Identity()
        self.w_norm = nn.BatchNorm2d(dim) if pre_norm else nn.Identity()

        self.activation = nn.GELU() if use_activation else nn.Identity()
        self.bn = nn.BatchNorm2d(dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.horizontal_conv(x)  # [B, C, H, 1]
        w = self.vertical_conv(x)    # [B, C, 1, W]

        h = self.h_norm(h)
        w = self.w_norm(w)

        if self.use_pos_prior:
            row_map = h * self.p_w       # [B, C, H, W]
            col_map = self.p_h * w       # [B, C, H, W]
            z = row_map * col_map        # [B, C, H, W]
        else:
            z = h * w                    # [B, C, H, W]

        z = self.activation(z)
        z = self.bn(z)
        return z

    def get_positional_prior_stats(self):
        if not self.use_pos_prior:
            return {
                "positional_prior": False,
                "channel_specific_pos": self.channel_specific_pos,
            }

        return {
            "p_h_mean": float(self.p_h.detach().cpu().mean().item()),
            "p_h_std": float(self.p_h.detach().cpu().std().item()),
            "p_w_mean": float(self.p_w.detach().cpu().mean().item()),
            "p_w_std": float(self.p_w.detach().cpu().std().item()),
            "channel_specific_pos": self.channel_specific_pos,
        }
