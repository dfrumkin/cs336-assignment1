import math
from pathlib import Path

import einx
import numpy as np
import torch
from hydra import main
from hydra.utils import instantiate
from jaxtyping import Int
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

import wandb
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.serialization import load_checkpoint, save_checkpoint


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: Int[np.ndarray, "*"], batch_size: int, context_length: int, device: str
) -> float:
    # No need for model.eval() because our model does not need it

    total_loss = 0
    total_tokens = 0
    n_tokens_block = batch_size * (context_length + 1)
    do_upcast = device == "mps"

    # total number of iterations (for tqdm progress bar)
    n_iters = math.ceil((dataset.shape[0] - context_length) / n_tokens_block)

    for start in tqdm(range(0, dataset.shape[0] - context_length, n_tokens_block), total=n_iters, desc="Validation"):
        available = min(dataset.shape[0] - start, n_tokens_block)
        batch_iter = available // (context_length + 1)
        n_tokens_iter = batch_iter * (context_length + 1)
        flat = dataset[start : start + n_tokens_iter]
        if do_upcast:
            # MPS does not work with uint16, or even int16.
            flat = flat.astype(np.int32)
        block_flat = torch.from_numpy(flat).to(device, non_blocking=True)
        block: Int[torch.Tensor, "b t"] = einx.rearrange(
            "(b t) -> b t",
            block_flat,
            b=batch_iter,
            t=context_length + 1,
        )  # type: ignore[assignment]
        inputs = block[:, :-1]
        logits = model(inputs)
        targets = block[:, 1:]

        loss = cross_entropy(logits, targets)
        total_loss += loss.item() * batch_iter * context_length
        total_tokens += batch_iter * context_length

    return total_loss / total_tokens


def calc_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))  # Avoid overflow in logging


@main(config_path="conf", config_name="train", version_base=None)
def run(cfg: DictConfig) -> None:
    # Get device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Instantiate and optionally compile the model
    model = instantiate(cfg.model, device=device)
    model.to(device)
    if cfg.compile:
        model: nn.Module = torch.compile(model, backend="aot_eager" if device == "mps" else "inductor")  # type: ignore[assignment]

    # Reduce precision for speed (works only on cuda)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # Compute num steps
    num_steps = (
        round(cfg.total_tokens_processed / (cfg.batch_size * cfg.context_length) / cfg.val_logging_freq)
        * cfg.val_logging_freq
    )

    # Standard parameters for a one-cycle cosine annealing
    warmup_iters = max(100, round(0.02 * num_steps))
    cosine_cycle_iters = num_steps - warmup_iters  # single decay, no restarts
    scheduler_fn = instantiate(cfg.scheduler, warmup_iters=warmup_iters, cosine_cycle_iters=cosine_cycle_iters)

    # Other instantiations
    get_batch_fn = instantiate(cfg.get_batch, device=device)
    grad_clipping_fn = instantiate(cfg.grad_clipping)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Load the dataset
    train_dataset = np.memmap(cfg.train_dataset, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(cfg.valid_dataset, dtype=np.uint16, mode="r")

    # Load the saved checkpoint
    if cfg.start_checkpoint is None:
        assert cfg.wandb_id is None
        start_step = 0
        resume = "never"
    else:
        assert cfg.wandb_id is not None
        start_step = load_checkpoint(cfg.start_checkpoint, model, optimizer) + 1
        resume = "must"

    # Log in to WnB using WANDB_API_KEY or ~/.netrc
    wandb.login()

    # Initialize WnB
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        id=cfg.wandb_id,
        resume=resume,
    )

    # Init the best result
    best_loss = float("inf")
    best_step = -1

    with tqdm(total=num_steps, initial=start_step, desc="Training") as pbar:
        for step in range(start_step, num_steps):
            # Get a batch
            inputs, targets = get_batch_fn(train_dataset)

            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)
            loss = cross_entropy(logits, targets)

            # Backward pass
            loss.backward()

            # Apply gradient clipping
            if grad_clipping_fn is not None:
                grad_clipping_fn(model.parameters())

            # Set the learning rate for the current step
            lr = scheduler_fn(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Optimizer step
            optimizer.step()

            # Log training loss and learning rate
            if (step + 1) % cfg.train_logging_freq == 0:
                wandb.log(
                    {"loss/train": loss.item(), "perplexity/train": calc_perplexity(loss.item()), "lr": lr},
                    step=step,
                )

            # Validation and checkpointing - less frequent
            if (step + 1) % cfg.val_logging_freq == 0:
                val_loss = evaluate(model, valid_dataset, cfg.batch_size, cfg.context_length, cfg.device)
                wandb.log(
                    {"loss/val": val_loss, "perplexity/val": calc_perplexity(val_loss)},
                    step=step,
                )

                path = Path(cfg.last_checkpoint)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, optimizer, step, cfg.last_checkpoint)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_step = step
                    save_checkpoint(model, optimizer, step, cfg.best_checkpoint)

            # Update tqdm
            pbar.update()

    # Graceful termination
    wandb.finish()
    print(f"Smallest loss / perplexity = {best_loss:.5f} / {calc_perplexity(best_loss):.5f}, on step {best_step:,}")


if __name__ == "__main__":
    run()
