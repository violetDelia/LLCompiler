import torch.nn as nn
import torch



class ReduceFusion1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rule = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x - 3
        x = x / 2
        x= torch.softmax(x,1)
        x = torch.matmul(x,x.transpose(-2, -1))
        x = x + x
        x = x - 3
        x = x / 2
        return x