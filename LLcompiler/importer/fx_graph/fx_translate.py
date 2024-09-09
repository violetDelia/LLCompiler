from xdsl.dialects.builtin import (
    IndexType,
    IntegerType,
    StringAttr,
    DenseIntOrFPElementsAttr,
    TensorType,
    DenseArrayBase,
    ShapedType,
    Signedness,
    BFloat16Type,
    i64,
    i32,
    i16,
    i1,
    f16,
    f32,
    f64,
    DYNAMIC_INDEX,
)
from ...dialect.llh import *
from datetime import datetime
import torch.nn
from torch._subclasses.fake_tensor import FakeTensor
from ...core.utility import run_time
import os
import numpy as np
import torch.fx


def torch_fake_tensor_translate(tensor: FakeTensor):
    ele_type = torch_dtype_translate(tensor.dtype)
    shape = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            shape.append(dim)
        if isinstance(dim, torch.SymInt):
            shape.append(DYNAMIC_INDEX)
    return TensorType(element_type=ele_type, shape=shape)


TORCH_DTYPE_TO_MLIR_TYPE = {
    torch.int64: i64,
    torch.int32: i32,
    torch.float16: f16,
    torch.float32: f32,
    torch.float64: f64,
    torch.bool: i1,
}


def torch_dtype_translate(dtype: torch.dtype):
    return TORCH_DTYPE_TO_MLIR_TYPE[dtype]




def torch_function_translate(
    node: torch.fx.node.Node, value_map: dict[str, list[SSAValue]]
) -> Operation:
    print(node.target)
    print(type(node.target))
