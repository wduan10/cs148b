import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., vocab_size)
    targets: (...,) integer indices

    returns:
        scalar loss (mean over all batch + sequence dims)
    """

    # Step 1: subtract max for numerical stability
    # (..., 1)
    max_logits = logits.max(dim=-1, keepdim=True).values

    # (..., vocab_size)
    logits_shifted = logits - max_logits

    # Step 2: compute logsumexp
    # (...,)
    logsumexp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))

    # Step 3: add back max (because we subtracted it earlier)
    # (...,)
    logsumexp = logsumexp + max_logits.squeeze(-1)

    # Step 4: gather logits at target indices
    # (...,)
    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    # Step 5: compute loss
    # ℓ = - o[y] + logsumexp
    loss = -target_logits + logsumexp

    # Step 6: average over all dimensions
    return loss.mean()