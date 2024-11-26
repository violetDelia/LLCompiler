import torch.nn as nn
import torch


class Decompose_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(224, 100)
        self.batch1 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 10)
        self.batch2 = nn.BatchNorm1d(10)
        self.rule = nn.ReLU()
        self.flaten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.batch1(x)
        x = self.rule(x)
        # x = self.linear2(x)
        # x = self.batch2(x)
        # x = self.rule(x)
        # x = self.flaten(x)
        return x
