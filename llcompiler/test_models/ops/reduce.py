import torch.nn as nn
import torch


class RecudeMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x2 = x.amax(x.shape[0]-1, keepdim=False)
        x1 = x.amax(x.shape[0]-1, keepdim=True)
        x1 = x1.squeeze(x.shape[0]-1) 
        x = x1+x2
        x = x.amax(-1, keepdim=True)
        return x
