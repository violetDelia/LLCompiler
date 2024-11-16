import torch.nn as nn
import torch


class Extract(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x[0]
        x = x[-2]
        return x
