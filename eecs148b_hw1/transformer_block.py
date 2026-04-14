import torch
import torch.nn as nn
from eecs148b_hw1 import layernorm, multihead_self_attention, positionwise_feedforward

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.ln1 = layernorm.LayerNorm(d_model, device=device, dtype=dtype)
        self.attn = multihead_self_attention.MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )

        self.ln2 = layernorm.LayerNorm(d_model, device=device, dtype=dtype)
        self.ffn = positionwise_feedforward.PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First sublayer: pre-norm attention + residual
        x = x + self.attn(self.ln1(x))

        # Second sublayer: pre-norm feedforward + residual
        x = x + self.ffn(self.ln2(x))

        return x