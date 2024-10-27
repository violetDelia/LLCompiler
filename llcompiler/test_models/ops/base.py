import torch.nn as nn
import torch


class ElementaryArithmetic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x + x
        x1 = x.reshape(x.shape[3], x.shape[2], x.shape[0], x.shape[1])
        x1 = x - 2
        x1 = x1 * 2
        x1 = x1 / 2
        x2 = x.reshape(x.shape[3], x.shape[0], x.shape[2], x.shape[1])
        x2 = x2 + x2
        x2 = x2.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = x2 - x1
        return x
