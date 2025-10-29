import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    dataset: Int[np.ndarray, "*"], batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "batch context"], Int[Tensor, "batch context"]]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (Int[np.ndarray, "*"]): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        tuple[Int[Tensor, "batch context"], Int[Tensor, "batch context"]]: The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    starts = np.random.randint(0, dataset.shape[0] - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length + 1)
    block = dataset[idx]
    batch = torch.from_numpy(block).to(device)
    x = batch[:, :context_length]
    y = batch[:, 1 : context_length + 1]
    return x, y
