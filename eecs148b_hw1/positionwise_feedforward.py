import torch
import torch.nn as nn
from eecs148b_hw1 import linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_layer1 = self.linear1(x)
        relu_out = torch.where(output_layer1 > 0, output_layer1, 0.0)
        output = self.linear2(relu_out)
        return output