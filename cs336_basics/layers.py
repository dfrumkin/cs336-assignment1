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
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, 0.0, sigma, -3.0, 3.0)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        """Applies the linear transformation to the input

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
            embedding_dim (int): Dimension of the embedding vectors, i.e. d_model
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
        nn.init.trunc_normal_(self.embeddings, 0.0, 1.0, -3.0, 3.0)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, "... d_model"]:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: (Int[Tensor, "..."]): Token IDs
        Returns:
            Float[Tensor, "... d_model"]: Token embedding vectors
        """
        return einx.get_at("[num_embeddings] d_model, ... -> ... d_model", self.embeddings, token_ids)


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Constructs an RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """Applies RMS normalization to the input tensor.

        Args:
            x (Float[Tensor, " ... d_model"]): Input tensor

        Returns:
            Float[Tensor, " ... d_model"]: Normalized output tensor of the same shape
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        result = (x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)) * self.gain

        # Return the result in the original dtype
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Constructs a SwiGLU FFN.

        Args:
            d_model (int): Hidden dimension of the model
            d_ff (int | None, optional): _description_. Defaults to approximately 8/3*d_model.
            device (torch.device | None, optional): _description_. Defaults to None.
            dtype (torch.dtype | None, optional): _description_. Defaults to None.
        """
        super().__init__()
        if d_ff is None:
            d_ff = int(((d_model * 8 / 3 + 32) // 64) * 64)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Applies SwiGLU to the input tensor

        Args:
            x (Float[Tensor, " ... d_model"]): Input tensor

        Returns:
            Float[Tensor, "... d_model"]: Output tensor
        """
        w1 = self.w1(x)
        return self.w2(w1 * torch.sigmoid(w1) * self.w3(x))
