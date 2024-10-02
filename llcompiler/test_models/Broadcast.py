import onnx.reference.ops.op_size
import torch.nn as nn
import torch


class Broadcast(nn.Module):
    def __init__(self):
        super(Broadcast, self).__init__()
        self.cf = nn.Linear(224, 1)

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3])
        x1 = torch.empty(x.shape[3])
        x = x + x1
        x2 = torch.empty(x.shape[2], x.shape[3])
        x = x + x2
        x3 = torch.empty(x.shape[3])
        x = x - x3
        x4 = torch.empty(x.shape[2], x.shape[3])
        x = x * x4
        x4 = torch.empty(x.shape[1], x.shape[2], x.shape[3])
        x = x / x4
        x = x + 1
        x = x + 2
        x = x * x
        x = x / 2
        return x
