import torch
import torch.nn as nn
from eecs148b_hw1 import linear, scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Standard separate projections
        self.q_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., T, D)
        *batch_dims, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {D}")

        # Project
        q = self.q_proj(x)   # (..., T, D)
        k = self.k_proj(x)   # (..., T, D)
        v = self.v_proj(x)   # (..., T, D)

        # Split into heads:
        # (..., T, D) -> (..., T, H, Hd) -> (..., H, T, Hd)
        q = q.reshape(*batch_dims, T, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*batch_dims, T, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*batch_dims, T, self.num_heads, self.head_dim).transpose(-3, -2)

        # Mask shape: (T, T), broadcasts across batch dims and heads
        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        out = scaled_dot_product_attention.scaled_dot_product_attention(
            q, k, v, mask=causal_mask
        )  # (..., H, T, Hd)

        # Recombine heads:
        # (..., H, T, Hd) -> (..., T, H, Hd) -> (..., T, D)
        out = out.transpose(-3, -2).contiguous().reshape(*batch_dims, T, D)

        return self.output_proj(out)