
import torch.nn as nn
import torch


class Conv2D_NCHW_FCHW(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            3,
            3,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=1,
            bias=False,
            dilation=1,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.conv(x)
        return x