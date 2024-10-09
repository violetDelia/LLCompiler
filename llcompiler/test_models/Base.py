import torch.nn as nn
import torch



class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            3, 10, stride=2, kernel_size=5, padding=2, dilation=5
        )
        self.conv_layer2 = nn.Conv2d(10, 3, kernel_size=5, padding=5, bias=True)
        self.batch = nn.BatchNorm2d(100)
        self.cf = nn.Linear(int((224 - 17) / 2 + 7), 2)

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[3], x.shape[2], x.shape[0], x.shape[1])
        x = x.reshape(x.shape[2], x.shape[3], x.shape[1], x.shape[0])
        # x = self.conv_layer1(x)
        x1 = x + x
        c = 2 + 2 * 5 / 3
        x = x / c
        x2 = x + x1 + x * x
        # x = self.conv_layer2(x2 + x1)
        # x = self.cf(x + x * x + x / 2)
        return x
