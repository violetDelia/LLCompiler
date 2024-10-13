# RUN: python %s| FileCheck %s
# CHECK: The calculations are correct.
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


if __name__ == "__main__":
    model = Div()
    input = torch.randn(3, 3, 100)
    check_model(model, input)
