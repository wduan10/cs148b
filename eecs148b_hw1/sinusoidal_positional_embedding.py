import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, device=None, dtype=None):
        super().__init__()
        
        # precompute positional embeddings
        # positions: (max_seq_len, 1)
        positions = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)

        # frequencies: (d_model // 2,)
        div_term = torch.arange(0, d_model, 2, device=device, dtype=dtype)
        div_term = torch.exp(-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) * div_term / d_model)

        # angles: (max_seq_len, d_model // 2)
        angles = positions * div_term

        # build embedding matrix
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        # register buffer
        self.register_buffer("positional_embeddings", pe, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        return self.positional_embeddings[token_positions]
