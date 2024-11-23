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
import torch.library
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Callable, Union

import sympy
from torch._inductor.lowering import (

    make_fallback,
    fallback_handler,

)
import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import SymInt
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols

from torch._functorch.aot_autograd import aot_function, aot_module, make_boxed_compiler
from torch._functorch.compile_utils import strip_overloads
from torch._functorch.partitioners import (
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(input_dim, 12)

    def forward(self, x):
        x = self.fc3(x)
        x = x + x
        x = x + x
        return x


def test_add(lhs, rhs):
    print("call test add")
    return lhs + rhs


def compiler_demo_inner(gm: torch.fx.GraphModule, inputs):
    gm.print_readable(False)
    gm.graph.print_tabular()
    for node in gm.graph.nodes:
        if node.target == aten.add.Tensor:
            fallback_handler(node.target)

    return gm.forward


aten = torch.ops.aten
default_decompositions = {aten.trace}
modle = SimpleNN(2)
modle_opt = torch.compile(
    model=modle,
    backend=aot_autograd(
        fw_compiler=compiler_demo_inner,
        decompositions=get_decompositions(default_decompositions),
    ),
    dynamic=True,
    fullgraph=False,
)
inputs = torch.ones(10, 2)
print( modle_opt(inputs))
