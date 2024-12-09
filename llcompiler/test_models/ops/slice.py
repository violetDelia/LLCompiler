import torch.nn as nn
import torch


class Extract(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x1 = x[1]
        x2 = x[-2]
        x3 = x1+x2
        x3 += x[0]
        return x3
    
class Slice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x1 = x[0:3]
        x2 = x[3:6]
        x3 = x1+x2
        return x3
