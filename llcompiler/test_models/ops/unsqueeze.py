import torch.nn as nn
import torch


class Unsqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(1)
        x = x1+x2
        return x
