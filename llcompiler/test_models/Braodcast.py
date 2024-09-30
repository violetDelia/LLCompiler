import onnx.reference.ops.op_size
import torch.nn as nn
import torch


class Braodcast(nn.Module):
    def __init__(self):
        super(Braodcast, self).__init__()
        self.cf = nn.Linear(224,1)

    def forward(self, x: torch.Tensor):
        y = self.cf(x)
        cc = x + y
        x = cc + 1
        x = x + 1
        x = x * x
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3])
        x = x - 1
        return x
