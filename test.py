import torch
import torch.fx.experimental
import torch.nn as nn
import torch.nn.functional as F
import torchvision
input1 = torch.empty(3,4,224,224)
input2 = torch.empty(224,1)
print(input1.matmul(input2).shape)


