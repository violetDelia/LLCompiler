# RUN: %PYTHON %s | FileCheck %s
import llcompiler.compiler as LLC
import llcompiler.core
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
import torch.fx
from llcompiler.core.utility import run_time
import onnx
import torchgen
import torch._dynamo
import torch.fx as fx
import xml.dom


class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == "__main__":

    model = Mnist()
    input = torch.randn((2, 1, 28, 28))
    compiler = LLC.LLCompiler(mode="inference",vebose_first_ir = True)
    model = torch.compile(
        model=model,
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    model(input)
    
    
