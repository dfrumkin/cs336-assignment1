import math

import einx
import torch
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Construct a linear transformation module.

        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, 0.0, sigma, -3, 3)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        """Apply the linear transformation to the input

        Args:
            x (Float[Tensor, "... in_features"]): Input tensor

        Returns:
            Float[Tensor, "... out_features"]: Output tensor
        """
        return einx.dot("... i, o i -> ... o", x, self.weights)
