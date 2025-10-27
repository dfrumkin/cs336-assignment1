import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model,
            optimizer, and iteration to.
    """
    obj = {"model": model.to("cpu").state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer | None, optional): Restore the state of this optimizer if provided.
            Defaults to None.
    Returns:
        int: the previously-serialized number of iterations.
    """
    device = next(model.parameters()).device
    obj = torch.load(src, weights_only=False, map_location=device)
    # If we saved from a compiled model, need to change names.
    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in obj["model"].items()}
    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
