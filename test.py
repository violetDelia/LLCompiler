import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.randn(3,7,7)
out =  F.max_pool2d_with_indices(x,[3,3])
print(out)


