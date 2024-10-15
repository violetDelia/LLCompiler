import torch
import torch.fx.experimental
import torch.nn as nn
import torch.nn.functional as F
import torchvision

input = torch.randn(2,64,9,17)
# target output size of 7x7 (square)
m = nn.MaxPool2d(7,2,(3,1),(2,1))
print(m(input).shape)

