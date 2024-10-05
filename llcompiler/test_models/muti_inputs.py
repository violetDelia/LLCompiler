import torch.nn as nn
import torch


class Multi_Add(nn.Module):
    def __init__(self):
        super(Multi_Add, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            3, 10, stride=2, kernel_size=5, padding=2, dilation=5
        )
        self.conv_layer2 = nn.Conv2d(10, 3, kernel_size=5, padding=5, bias=True)
        self.batch = nn.BatchNorm2d(100)
        self.cf = nn.Linear(int((224 - 17) / 2 + 7), 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + y
        return x
