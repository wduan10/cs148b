import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtract max for numerical stability (keepdim=True for broadcasting)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    
    # exponentiate shifted values
    exp_x = torch.exp(x - x_max)
    
    # normalize
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    return exp_x / sum_exp