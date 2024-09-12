import torch
import torch.nn as nn

data1 = torch.randn(2, 3)


linear = torch.nn.Linear(3, 2)
print(data1.shape)
print(linear.weight.T.shape)
print(
    torch.addmm(
        linear.bias.broadcast_to(linear.bias.shape[0], linear.bias.shape[0]),
        data1,
        linear.weight.T,
    )
)
print(linear(data1))
print(
    linear.bias.broadcast_to(linear.bias.shape[0], linear.bias.shape[0])
    + data1.matmul(linear.weight.T)
)
