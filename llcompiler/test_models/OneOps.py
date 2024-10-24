import torch.nn as nn
import torch


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x+1
        x = x*x
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


class Conv_NCHW_FCHW(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            3,
            1,
            kernel_size=7,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x

class Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(x)
        return x