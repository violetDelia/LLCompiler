import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.randn(3,7,7)
out =  F.max_pool2d_with_indices(x,[3,3])
print(out)
import sys 
sys.path.append("/home/lfr/LLCompiler/build/lib.linux-x86_64-3.10")
import example as example
print( example.add(3,4))

