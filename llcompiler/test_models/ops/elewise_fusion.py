import torch.nn as nn
import torch


class ElewiseFusion1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rule = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x - 3
        x = x / 2
        x_max = self.rule(x)
        x = x_max * x
        return x
