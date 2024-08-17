import torch
from torchvision import models
import torch.fx as fx
import onnx


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


if __name__ == "__main__":
    # model = models.resnet18(pretrained=True)
    # x = torch.Tensor(1,3,224,224)
    # traced_cell = torch.jit.trace(model, x)
    # print(traced_cell.graph)

    my_cell = MyCell()
    x, h = torch.rand(3, 4), torch.rand(3, 4)
    traced_cell = torch.jit.trace(my_cell, (x, h))
    print(traced_cell.graph)
    traced_cell(x, h)
