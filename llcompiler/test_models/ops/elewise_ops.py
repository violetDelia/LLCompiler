import torch.nn as nn
import torch


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x + 3
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


class Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(x)
        return x


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x.abs()
        x = torch.abs(x)
        return x


class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x.sqrt()
        return x


class EQ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x == x
        return x == 2


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x.exp()
        return x

class Drop(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout()

    def forward(self, x: torch.Tensor):
        x = self.drop(x)
        return x