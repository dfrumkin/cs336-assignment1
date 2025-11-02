import math

import einx
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from cs336_basics.nn_utils import softmax


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
        # For tied weights
        # std=1.0 / math.sqrt(embedding_dim)
        std = 1.0
        nn.init.trunc_normal_(self.embeddings, 0.0, std, -3.0, 3.0)

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

        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x * norm * self.gain.to(torch.float32)

        # Return the result in the original dtype
        return result.to(in_dtype)


def silu(x: Float[Tensor, "... d_ff"]) -> Float[Tensor, "... d_ff"]:
    """Applies SiLU to the input tensor

    Args:
        x (Float[Tensor, " ... d_ff"]): Input tensor

    Returns:
        Float[Tensor, "... d_ff"]: Output tensor
    """
    return x * torch.sigmoid(x)


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
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        if d_ff is None:
            d_ff = int(((d_model * 8 / 3 + 32) // 64) * 64)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Applies the SwiGLU FFN to the input tensor

        Args:
            x (Float[Tensor, " ... d_model"]): Input tensor

        Returns:
            Float[Tensor, "... d_model"]: Output tensor
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Constructs a SiLU FFN.

        Args:
            d_model (int): Hidden dimension of the model
            d_ff (int | None, optional): _description_. Defaults to approximately 8/3*d_model.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        if d_ff is None:
            d_ff = int(((d_model * 4 + 32) // 64) * 64)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Applies the SiLU FFN to the input tensor

        Args:
            x (Float[Tensor, " ... d_model"]): Input tensor

        Returns:
            Float[Tensor, "... d_model"]: Output tensor
        """
        return self.w2(silu(self.w1(x)))


class RotaryPositionEmbedding(nn.Module):
    cos_sin: Float[Tensor, "l n 2"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Constructs the RoPE module.

        Args:
            theta (float): Theta value for the RoPE
            d_k (int): Dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()

        # Frequencies within a token
        n = torch.arange(d_k // 2, device=device)
        inv_freq = theta ** (-2 * n / d_k)

        # Token position
        pos = torch.arange(max_seq_len, device=device)

        # Outer product
        angles = einx.multiply("l, n -> l n", pos, inv_freq)

        # Compute cosines and sines
        cos_sin = torch.stack([angles.cos(), angles.sin()], dim=-1)

        # Register as buffer (not trainable and not part of state_dict)
        self.register_buffer("cos_sin", cos_sin.to(dtype), persistent=False)

    def forward(
        self, x: Float[Tensor, " ... head seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... head seq_len d_k"]:
        """Applies RoPE to the input tensor.

        Args:
            x (Float[Tensor, " ... head seq_len d_k"]): Input tensor
            token_positions (Int[Tensor, " ... seq_len"]): Token positions.

        Returns:
            Float[Tensor, " ... seq_len d_k"]: Output tensor
        """
        # Select relevant rotations
        cos_sin = einx.get_at("[l] n p, ... i -> ... i n p", self.cos_sin, token_positions, p=2)

        # Split into pairs (n == d_k / 2)
        x2: Float[Tensor, "... seq_len n 2"] = einx.rearrange("... l (n p) -> ... l n p", x, p=2)  # type: ignore[assignment]

        # Multiply by cosines and sines (outer product)
        prods = einx.multiply("... h l n p, ... l n q -> ... h l n p q", x2, cos_sin, p=2, q=2)

        # Combine into rotated components
        y_even = prods[..., 0, 0] - prods[..., 1, 1]
        y_odd = prods[..., 1, 0] + prods[..., 0, 1]

        # Merge rotated vectors
        y: Float[Tensor, "... seq_len d_k"] = einx.rearrange(
            "... l n p -> ... l (n p)", torch.stack([y_even, y_odd], dim=-1), p=2
        )  # type: ignore[assignment]

        return y


def scaled_dot_product_attention(
    queries: Float[Tensor, " ... seq_len d_k"],
    keys: Float[Tensor, " ... seq_len d_k"],
    values: Float[Tensor, " ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, " ... seq_len d_v"]:
    """Computes the scaled dot-product attention.
    Args:
        queries (Float[Tensor, " ... seq_len d_k"]): Queries
        keys (Float[Tensor, " ... seq_len d_k"]): Keys
        values (Float[Tensor, " ... seq_len d_v"]): Values
        mask (Bool[Tensor, "seq_len seq_len"] | None, optional): Attention mask.  Defaults to None.

    Returns:
        Float[Tensor, " ... seq_len d_v"]: Attention output
    """
    scores = einx.dot("b ... i d, b ... j d -> b ... i j", queries, keys) / math.sqrt(keys.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    scores = softmax(scores, dim=-1).to(values.dtype)
    return einx.dot("... i j, ... j d -> ... i d", scores, values)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Constructs causal multi-head self-attention.

        Args:
            d_model (int): Hidden dimension of the model
            num_heads (int): Number of attention heads
            rope (RotaryPositionEmbedding | None, optional): RoPE module. Defaults to None.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_model // num_heads
        self.qkv = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.out = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, " ... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Applies multi-head self-attention to the input tensor.

        Args:
            x (Float[Tensor, "... seq_len d_model"]): Input tensor
            token_positions (Int[Tensor, " ... seq_len"] | None, optional): Token positions.  Default to 0..seq_len-1.
        Returns:
            Float[Tensor, "... seq_len d_model"]: Output tensor
        """
        seq_len = x.shape[-2]
        qkv = self.qkv(x)
        qkv3: Float[Tensor, "3 ... num_heads seq_len d_kv"] = einx.rearrange(
            "... t (three h d) -> three ... h t d", qkv, three=3, h=self.num_heads, t=seq_len, d=self.d_kv
        )  # type: ignore
        q, k, v = qkv3.unbind(0)

        if self.rope:
            token_positions = (
                torch.arange(seq_len, device=q.device) if token_positions is None else token_positions.squeeze()
            )
            if len(token_positions.shape) == 1:
                # Same for every sequence in the batch
                token_positions = token_positions.expand(x.shape[0], -1)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        attention_out = scaled_dot_product_attention(q, k, v, mask)
        concat = einx.rearrange("... h t d -> ... t (h d)", attention_out, t=seq_len, h=self.num_heads, d=self.d_kv)
        return self.out(concat)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Constructs the pre-norm Transformer block.

        Args:
            d_model (int): Dimensionality of the Transformer block inputs
            num_heads (int): Number of heads to use in multi-head self-attention
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer
            rope (RotaryPositionEmbedding | None, optional): RoPE module. Defaults to None.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        # SiLU - remember to adjust d_ff
        # self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, " ... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Applies the transformer block to the input tensor.

        Args:
            x (Float[Tensor, "... seq_len d_model"]): Input tensor
            token_positions (Int[Tensor, " ... seq_len"] | None, optional): Token positions.  Default to 0..seq_len-1.
        Returns:
            Float[Tensor, "... seq_len d_model"]: Output tensor
        """
        # Pre-norm
        y = x + self.attn(self.ln1(x), token_positions)
        y = y + self.ffn(self.ln2(y))
        # Post-norm
        # y = self.ln1(x + self.attn(x, token_positions))
        # y = self.ln2(y + self.ffn(y))
        return y


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        r"""Constructs the Transformer language mode.

        Args:
            d_model (int): The dimensionality of the model embeddings and sublayer outputs
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer
            rope_theta (float): The RoPE $\Theta$ parameter
            vocab_size (int): The number of unique items in the output vocabulary to be predicted
            context_length (int): The maximum number of tokens to process at once
            num_layers (int): The number of Transformer layers to use
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        d_k = d_model // num_heads

        rope = RotaryPositionEmbedding(rope_theta, d_k, context_length, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        # Pre-norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        # Weight tying
        # self.lm_head.weights = self.embedding.embeddings

    def forward(self, in_indices: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """Applies the transformer to token indices

        Args:
            in_indices: (Int[Tensor, "batch_size seq_len"]): Input token indices
        Returns:
            Float[Tensor, "batch_size seq_len vocab_size"]: For each input token returns
                raw predicted logits over the vocabulary.
        """
        x = self.embedding(in_indices)

        for layer in self.layers:
            x = layer(x)

        # Return in the model datatype
        # Pre-norm
        return self.lm_head(self.ln_final(x))
        # Post-norm
        # return self.lm_head(x)
