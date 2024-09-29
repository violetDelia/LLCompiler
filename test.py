import torch
import torch.fx.experimental
import torch.nn as nn
import torch.nn.functional as F
import torchvision
filters = torch.randn(64,3,7,7)
inputs = torch.randn(1, 3,224 , 224)
print(F.conv2d(inputs, filters, padding=1).shape)
filters = filters.permute(0,2,3,1)
inputs = inputs.permute(0,2,3,1)
print(F.conv2d(inputs, filters, padding=1).shape)


