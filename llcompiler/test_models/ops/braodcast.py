import torch.nn as nn
import torch


class Braodcast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x1 = x.reshape(1, x.shape[0], x.shape[1])
        x1 = x + x1
        x2 = x.reshape(x.shape[0], 1, x.shape[1])
        x2 = x + x2
        x3 = x.reshape(x.shape[0], 1, x.shape[1])
        x = x1 + x2
        # x4 = x[0]
        # x5 = x4[0]
        # x = x * x5
        return x
