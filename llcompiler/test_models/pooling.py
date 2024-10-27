import torch.nn as nn
import torch


class MaxPool2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(
            5,
            1,
        )

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        return x
