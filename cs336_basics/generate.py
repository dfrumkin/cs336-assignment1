import torch
from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.console import Console

from cs336_basics.model import Transformer
from cs336_basics.nn_utils import softmax
from cs336_basics.serialization import load_checkpoint
from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"


@torch.inference_mode()
def decode(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    end_token = tokenizer.encode(END_OF_TEXT)[0]
    tokens = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        seq_start = max(len(tokens) - context_length, 0)
        in_tokens = torch.tensor(tokens[seq_start : len(tokens)])[None]
        logits = model(in_tokens)[0, -1, :]
        probs = softmax(logits, dim=0, temp=temperature)

        sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        cutoff = torch.searchsorted(cumsum, top_p) + 1  # capture >= top_p
        trimmed_probs = sorted_probs[:cutoff]
        trimmed_idxs = sorted_idxs[:cutoff]
        trimmed_probs /= trimmed_probs.sum()

        next_token: int = trimmed_idxs[torch.multinomial(trimmed_probs, 1).item()].item()  # type: ignore[assignment]
        tokens.append(next_token)

        if next_token == end_token:
            break

    return tokenizer.decode(tokens[len(prompt) :])


@main(config_path="conf", config_name="generate", version_base=None)
def run(cfg: DictConfig) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(cfg.model, device=device)
    load_checkpoint(cfg.checkpoint, model)
    model.to(device)
    # model.eval() is not needed

    continuation = decode(model, tokenizer, prompt=cfg.prompt, context_length=cfg.model.context_length, **cfg.decode)
    console = Console()
    console.print(f"[dim]{cfg.prompt}[/dim][bold green]{continuation}[/bold green]")


if __name__ == "__main__":
    run()
