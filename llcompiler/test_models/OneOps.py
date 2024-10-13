import torch.nn as nn
import torch


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x + 2
        return x


class Sub(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x - x
        x = x - 2
        return x


class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x / x
        x = x / 2
        return x


class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x * x
        x = x * 2
        return x
