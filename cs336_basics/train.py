import math
import random
import warnings
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

warnings.filterwarnings(
    "ignore",
    message="Skipping serialization of skipfiles_inline_module_allowlist",
)
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
)


def fix_random(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Int[np.ndarray, "*"],
    batch_size: int,
    context_length: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> float:
    # No need for model.eval() because our model does not need it

    total_loss = 0
    total_tokens = 0
    n_tokens_block = batch_size * (context_length + 1)

    # total number of iterations (for tqdm progress bar)
    n_iters = math.ceil((dataset.shape[0] - context_length) / n_tokens_block)

    for start in tqdm(range(0, dataset.shape[0] - context_length, n_tokens_block), total=n_iters, desc="Validation"):
        available = min(dataset.shape[0] - start, n_tokens_block)
        batch_iter = available // (context_length + 1)
        n_tokens_iter = batch_iter * (context_length + 1)
        flat = dataset[start : start + n_tokens_iter].astype(np.int32)
        block_flat = torch.from_numpy(flat).to(device, non_blocking=True)
        block: Int[torch.Tensor, "b t"] = einx.rearrange(
            "(b t) -> b t",
            block_flat,
            b=batch_iter,
            t=context_length + 1,
        )  # type: ignore[assignment]
        inputs = block[:, :-1]
        targets = block[:, 1:]

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(inputs)
        loss = cross_entropy(logits, targets)

        total_loss += loss.item() * batch_iter * context_length
        total_tokens += batch_iter * context_length

    return total_loss / total_tokens


def calc_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))  # Avoid overflow in logging


def test_params(model, msg):
    bad_params = []
    bad_grads = []

    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad_params.append(n)
        if p.grad is not None:
            gmax = p.grad.abs().max()
            if not torch.isfinite(gmax).all():
                bad_grads.append(n)

    if bad_params:
        print(f"{msg}: NaNs/Infs found in parameters:", bad_params)
        raise AssertionError

    if bad_grads:
        print(f"{msg}: NaNs/Infs found in gradients:", bad_grads)
        raise AssertionError

    if not (bad_params or bad_grads):
        print(f"{msg}: Parameters/gradients are finite.")


@main(config_path="conf", config_name="train", version_base=None)
def run(cfg: DictConfig) -> None:
    # Fix randomness
    fix_random()

    # Get device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Reduce precision for speed (works only on cuda)
    dtype = None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if cfg.mixed_precision:
            dtype = torch.bfloat16

    # Instantiate and optionally compile the model
    model = instantiate(cfg.model, device=device, dtype=dtype)
    model.to(device)
    if cfg.compile:
        # Got numerical problems when compiling on CUDA with mixed precision... :(
        backend = "aot_eager" if device.type == "mps" or cfg.mixed_precision else "inductor"
        model: nn.Module = torch.compile(model, backend=backend)  # type: ignore[assignment]

    # Compute num steps
    num_steps = (
        round(cfg.total_tokens_processed / (cfg.batch_size * cfg.context_length) / cfg.val_logging_freq)
        * cfg.val_logging_freq
    )

    # Standard parameters for a one-cycle cosine annealing
    warmup_iters = max(100, round(0.02 * num_steps))
    cosine_cycle_iters = num_steps - warmup_iters  # single decay, no restarts
    scheduler_fn = instantiate(
        cfg.scheduler,
        warmup_iters=warmup_iters,
        max_learning_rate=cfg.lr,
        min_learning_rate=0.01 * cfg.lr,
        cosine_cycle_iters=cosine_cycle_iters,
    )

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

    # Initializations
    total_loss = 0.0
    best_loss = float("inf")
    best_step = -1
    run_name = cfg.run_name  # Or smth else for sweeps
    run_ckpt_dir = Path(cfg.checkpoint_dir) / run_name
    last_checkpoint = run_ckpt_dir / "last.ckpt"
    best_checkpoint = run_ckpt_dir / "best.ckpt"

    # Initialize WnB
    wandb.init(
        project=cfg.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        id=cfg.wandb_id,
        resume=resume,
    )

    with tqdm(total=num_steps, initial=start_step, desc="Training") as pbar:
        for step in range(start_step, num_steps):
            # Get a batch
            inputs, targets = get_batch_fn(train_dataset)

            # Reset the gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()

            # Are all parameters finite?
            # test_params(model, f"forward {step}")

            # Backward pass
            loss.backward()

            # Are all parameters finite?
            # test_params(model, f"backward {step}")

            # Apply gradient clipping
            if grad_clipping_fn is not None:
                grad_clipping_fn(model.parameters())

            # Are all parameters finite?
            # test_params(model, f"clipping {step}")

            # Set the learning rate for the current step
            lr = scheduler_fn(step + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Optimizer step
            optimizer.step()

            # Are all parameters finite?
            # test_params(model, f"optimizer {step}")

            # Log training loss and learning rate
            if (step + 1) % cfg.train_logging_freq == 0:
                avg_loss = total_loss / cfg.train_logging_freq
                total_loss = 0.0
                wandb.log(
                    {
                        "loss/train": avg_loss,
                        "perplexity/train": calc_perplexity(avg_loss),
                        "lr": lr,
                    },
                    step=step,
                )

            # Validation and checkpointing - less frequent
            if (step + 1) % cfg.val_logging_freq == 0 or step + 1 == num_steps:
                val_loss = evaluate(model, valid_dataset, cfg.batch_size, cfg.context_length, device, dtype)
                wandb.log(
                    {"loss/val": val_loss, "perplexity/val": calc_perplexity(val_loss)},
                    step=step,
                )

                run_ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, optimizer, step, last_checkpoint)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_step = step
                    save_checkpoint(model, optimizer, step, best_checkpoint)

            # Update tqdm
            pbar.update()

    # Graceful termination
    wandb.finish()
    print(f"Smallest loss / perplexity = {best_loss:.5f} / {calc_perplexity(best_loss):.5f}, on step {best_step:,}")


if __name__ == "__main__":
    run()
