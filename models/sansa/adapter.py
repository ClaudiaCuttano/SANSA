# --------------------------------------------------------
# References:
# https://github.com/ShoufaChen/AdaptFormer
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    AdaptFormer (pre-LayerNorm, linear down->nonlinearity->linear up).

    The module projects features from `in_channel` to a lower-dimensional bottleneck,
    applies a ReLU, then projects back and scales the output. We initialize the
    up-projection weights to zero so the adapter starts as an identity (no effect)
    when used in a residual path upstream.

    Args:
        in_channel (int): Input feature dimension.
        bottleneck (int): Bottleneck (down-projected) dimension.
        adapter_scalar (float): Scalar applied to the up-projected output.

    Forward:
        x (Tensor): [..., in_channel]
        returns (Tensor): same shape as x (adapter update to be combined by caller)
    """

    def __init__(self, in_channel: int, bottleneck: int = 64, adapter_scalar: float = 0.1) -> None:
        super().__init__()
        self.n_embd = in_channel
        self.down_size = bottleneck
        self.scale = adapter_scalar

        # Pre-norm
        self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # Bottleneck MLP
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # Initialization: kaiming for down, zeros for up (so adapter starts "off")
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm → Linear↓ → ReLU → Linear↑ → scale.
        Returns:
            Tensor with same shape as x.
        """
        x = self.adapter_layer_norm_before(x)
        x = self.down_proj(x)
        x = self.non_linear_func(x)
        x = self.up_proj(x)
        return x * self.scale
