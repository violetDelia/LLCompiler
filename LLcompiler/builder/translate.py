from xdsl.dialects.builtin import (
    IndexType,
    IntegerType,
    StringAttr,
    DenseIntOrFPElementsAttr,
    TensorType,
    ShapedType,
    i64,
    i32,
    i16,
    i1,
    f16,
    f32,
    f64,
    DYNAMIC_INDEX,
)
import torch.nn
from torch._subclasses.fake_tensor import FakeTensor
from ..core.utility import run_time
import os
import numpy as np
import torch.fx


def torch_module_translate(node: torch.fx.node.Node, value_map: dict):
    module_stack = node.meta['nn_module_stack']
    target = node.target
    print(module_stack)
    if module_stack[target][1] is torch.nn.modules.conv.Conv2d:
        print(node)
    raise NotImplementedError


def torch_tensor_translate(tensor: FakeTensor):
    ele_type = torch_dtype_translate(tensor.dtype)
    shape = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            shape.append(dim)
        if isinstance(dim, torch.SymInt):
            shape.append(DYNAMIC_INDEX)
    return TensorType(element_type=ele_type, shape=shape)


def torch_dtype_translate(dtype: torch.dtype):
    match dtype:
        case torch.int64:
            return i64
        case torch.int32:
            return i32
        case torch.float16:
            return f16
        case torch.float32:
            return f32
        case torch.float64:
            return f64
        case torch.bool:
            return i1
        case _:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")
