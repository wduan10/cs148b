import numpy as np
import torch


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    x: numpy array of shape (n,) containing token IDs
    returns:
        inputs:  (batch_size, context_length)
        targets: (batch_size, context_length)
    """

    n = len(x)

    # sample random starting indices
    # ensure we have room for context_length + 1 tokens
    starts = np.random.randint(0, n - context_length, size=batch_size)

    # build batches
    inputs = []
    targets = []

    for i in starts:
        chunk = x[i : i + context_length + 1]  # length m+1

        inputs.append(chunk[:-1])   # first m tokens
        targets.append(chunk[1:])   # next m tokens

    # convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets