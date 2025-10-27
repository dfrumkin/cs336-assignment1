import einx
import torch
from jaxtyping import Float, Int
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int, temp: float | None = None) -> Float[Tensor, "..."]:
    """Applies softmax along the given dimension.

    Args:
        x (Float[Tensor, "..."]): Input tensor
        dim (int): Dimension
        temp (float | None, optional): Temperature.  Defaults to None.

    Returns:
        Float[Tensor, "..."]: Output tensor - always as float32
    """
    # Cast the input to float32 for exponentiation
    x = x.to(torch.float32)

    if temp is not None:
        x = x / temp

    exps = torch.exp(x - x.amax(dim, keepdim=True))

    # Return in float32 for multiplication by values (stability - aggregating over a long sequence)
    return exps / exps.sum(dim, keepdim=True)


def cross_entropy(logits: Float[Tensor, "... num_classes"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    """Computes cross-entropy loss (log-loss).

    Args:
        logits (Float[Tensor, "... num_classes"]): Logits
        targets (Int[Tensor, "..."]): Target labels

    Returns:
        Float[Tensor, ""]: Mean log-loss
    """
    # Cast the input to float32 for exponentiation
    logits = logits.to(torch.float32)

    # Predictions at targets
    z_y = einx.get_at("... [c], ... -> ...", logits, targets)

    # Log-sum-exp
    max_vals = logits.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.exp(logits - max_vals).sum(dim=-1)) + max_vals.squeeze(-1)

    return (log_sum_exp - z_y).mean()
