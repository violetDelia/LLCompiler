import torch.nn as nn
import torch
class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = torch.matmul(x,x.transpose(-2, -1))
        return x
