import math
from collections.abc import Callable

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

    def step(self, closure: Callable[[], Float[Tensor, ""]] | None = None) -> Float[Tensor, ""] | None:
        """Performs an optimization step

        Args:
            closure (Callable[[], Float[Tensor, ""]] | None, optional): Closure for computing the loss.
                Defaults to None.
        Returns:
            Float[Tensor, ""] | None: Loss if closure is provided
        """
        loss = None if closure is None else closure()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    lr = group["lr"]
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    weight_decay = group["weight_decay"]

                    grad = p.grad  # Gradient of loss with respect to p
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

                    # Adjust the learning rate
                    lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                    # Update the parameters and apply weight decay
                    p.addcdiv_(m, torch.sqrt(v) + eps, value=-lr_t).mul_(1 - lr * weight_decay)

                    state["t"] = t + 1  # Increment iteration number.
        return loss
