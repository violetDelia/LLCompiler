import torch.nn as nn
import torch


class Where(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre, x: torch.Tensor, y):
        x = x.where(pre, y)
        return x
