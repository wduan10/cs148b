import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # compute mean and variance over last dimension d_model
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        result = x_hat * self.g + self.b
        return result.to(in_dtype)