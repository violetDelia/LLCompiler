import torch.nn as nn
import torch


class Unsqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        return x
