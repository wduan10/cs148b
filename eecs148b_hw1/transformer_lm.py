import torch
import torch.nn as nn

from eecs148b_hw1 import (
    embedding,
    sinusoidal_positional_embedding,
    transformer_block,
    layernorm,
    linear,
)


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = embedding.Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.pos_embeddings = sinusoidal_positional_embedding.SinusoidalPositionalEmbedding(
            context_length,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList([
            transformer_block.TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = layernorm.LayerNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        # LM head: hidden state -> vocab logits
        self.lm_head = linear.Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (..., seq_len)
        returns: (..., seq_len, vocab_size)
        """
        *batch_dims, seq_len = token_ids.shape

        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.context_length}"
            )

        # token embeddings: (..., seq_len, d_model)
        tok_emb = self.token_embeddings(token_ids)

        # positions: (seq_len,)
        positions = torch.arange(seq_len, device=token_ids.device)

        # positional embeddings: (seq_len, d_model)
        pos_emb = self.pos_embeddings(positions)

        # broadcast add -> (..., seq_len, d_model)
        # x = tok_emb + pos_emb
        x = tok_emb # ablation: without positional embedding

        # transformer stack
        for layer in self.layers:
            x = layer(x)

        # final layer norm
        x = self.ln_final(x)
        # in a layernorm ablation ablation, I comment the above line out

        # vocab logits: (..., seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits