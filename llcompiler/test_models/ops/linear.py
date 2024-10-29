import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            100000,
            10,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x
