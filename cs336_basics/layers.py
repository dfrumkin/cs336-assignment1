import math

import einx
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Constructs a linear transformation module.

        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, 0.0, sigma, -3.0, 3.0)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        """Apply the linear transformation to the input

        Args:
            x (Float[Tensor, "... in_features"]): Input tensor

        Returns:
            Float[Tensor, "... out_features"]: Output tensor
        """
        return einx.dot("... i, o i -> ... o", x, self.weights)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Constructs an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (torch.device | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
        nn.init.trunc_normal_(self.embeddings, 0.0, 1.0, -3.0, 3.0)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, "... embedding_dim"]:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: (Int[Tensor, "..."]): Token IDs
        Returns:
            Float[Tensor, "... embedding_dim"]: Token embedding vectors
        """
        return einx.get_at("[num_embeddings] embedding_dim, ... -> ... embedding_dim", self.embeddings, token_ids)
