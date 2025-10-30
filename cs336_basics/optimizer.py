import math
from collections.abc import Callable, Iterable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2) -> None:
        """Constructs an AdamW optimizer.

        Args:
            params (torch.optim.optimizer.ParamsT): Model parameters
            lr (_type_, optional): Learning rate. Defaults to 1e-3.
            betas (tuple, optional): Coefficients for computing the running averages. Defaults to (0.9, 0.999).
            eps (_type_, optional): Epsilon for numerical stability. Defaults to 1e-8.
            weight_decay (_type_, optional): Weight decay coefficient. Defaults to 1e-2.

        Raises:
            ValueError: Raised if coefficients are out of range
        """
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Float[Tensor, ""]] | None = None) -> Float[Tensor, ""] | None:
        """Performs an optimization step.

        Args:
            closure (Callable[[], Float[Tensor, ""]] | None, optional): Closure for computing the loss.
                Defaults to None.
        Returns:
            Float[Tensor, ""] | None: Loss if closure is provided
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach().to(torch.float32)  # Gradient of loss with respect to p
                state = self.state[p]  # Get state associated with p.

                if len(state) == 0:
                    state["t"] = 1  # Iteration number starting from 1
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)  # First moment
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)  # Second moment

                t = state["t"]
                m = state["m"]
                v = state["v"]

                # Moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias-corrected step size (fp32 on same device)
                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                lr_t = lr * (bc2**0.5) / bc1

                # update in fp32
                denom = v.clamp(min=0).sqrt().add_(eps)
                upd32 = m / denom

                # decoupled weight decay
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)  # in param dtype

                # apply update (cast once to param dtype)
                p.add_(upd32.to(dtype=p.dtype), alpha=-lr_t)

                state["t"] = t + 1  # Increment iteration number.
        return loss


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        float: Learning rate at the given iteration under the specified schedule.
    """
    return (
        it / warmup_iters * max_learning_rate
        if it < warmup_iters
        else min_learning_rate
        + 0.5
        * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
        * (max_learning_rate - min_learning_rate)
        if warmup_iters <= it <= cosine_cycle_iters
        else min_learning_rate
    )


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # Compute global L2 norm using torch.linalg.vector_norm for stability
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads]))

    scale = max_l2_norm / (total_norm + eps)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)
