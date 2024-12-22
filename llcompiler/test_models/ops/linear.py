import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(
            100000,
            10,
            bias=False,
        )
        self.linear2 = nn.Linear(
            10,
            100,
            bias=False,
        )
        self.linear3 = nn.Linear(
            100,
            10,
            bias=False,
        )
        self.linear1.train(False)
        self.linear2.train(False)
        self.linear3.train(False)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
