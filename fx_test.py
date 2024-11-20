import llcompiler.compiler as LLC
import os.path
import subprocess
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
import os
from llcompiler.test_models import *


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compiler_demo_inner(gm: torch.fx.GraphModule, inputs):
    gm.print_readable(False)
    gm.graph.print_tabular()
    return gm.forward

    

modle = SimpleNN(2)
modle_opt = torch.compile(
    model=modle,
    backend=aot_autograd(fw_compiler = compiler_demo_inner),
    dynamic=True,
    fullgraph=True,
)
inputs = torch.randn(10, 2)
modle_opt(inputs)
