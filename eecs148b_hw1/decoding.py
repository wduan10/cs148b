import torch


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    probs: (vocab_size,)
    returns: sampled token id, shape ()
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1].")

    # Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep smallest set whose cumulative mass >= top_p
    keep_mask = cumulative_probs <= top_p

    # Ensure at least one token is kept
    keep_mask[0] = True

    filtered_probs = sorted_probs * keep_mask
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_idx_in_sorted = torch.multinomial(filtered_probs, num_samples=1)
    sampled_token = sorted_indices[sampled_idx_in_sorted]

    return sampled_token.squeeze(0)


@torch.no_grad()
def decode(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_token_id: int | None = None,
):
    """
    Generate tokens autoregressively from a language model.

    Args:
        model:
            A Transformer LM that maps token ids of shape (B, T) to logits of shape (B, T, vocab_size).
        prompt:
            Tensor of token ids with shape (T,) or (1, T).
        max_new_tokens:
            Maximum number of tokens to generate.
        temperature:
            Softmax temperature. If 0, use greedy decoding.
        top_p:
            Nucleus sampling threshold in (0, 1]. If 1.0, no top-p truncation.
        end_of_text_token_id:
            Stop generation if this token is produced.

    Returns:
        Tensor of token ids containing prompt + generated tokens, shape (T_total,)
    """
    model.eval()

    if prompt.dim() == 1:
        tokens = prompt.unsqueeze(0)  # (1, T)
    elif prompt.dim() == 2 and prompt.shape[0] == 1:
        tokens = prompt
    else:
        raise ValueError("prompt must have shape (T,) or (1, T)")

    device = next(model.parameters()).device
    tokens = tokens.to(device=device, dtype=torch.long)

    # Infer context length if present
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        # Crop to context window if needed
        if context_length is not None and tokens.shape[1] > context_length:
            input_tokens = tokens[:, -context_length:]
        else:
            input_tokens = tokens

        logits = model(input_tokens)           # (1, T, vocab_size)
        next_token_logits = logits[:, -1, :]   # (1, vocab_size)
        next_token_logits = next_token_logits.squeeze(0)  # (vocab_size,)

        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            if temperature < 0:
                raise ValueError("temperature must be >= 0")

            scaled_logits = next_token_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            if top_p < 1.0:
                next_token = sample_top_p(probs, top_p).unsqueeze(0)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

        if end_of_text_token_id is not None and next_token.item() == end_of_text_token_id:
            break

    return tokens.squeeze(0)
