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
from ..dialect.llh import *
from datetime import datetime
import torch.nn
from torch._subclasses.fake_tensor import FakeTensor
from ..core.utility import run_time
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


def torch_conv_convert(
    node: torch.fx.node.Node, value_map: dict, module: torch.nn.modules.conv._ConvNd
):
    tensor = node.meta["example_value"]
    result_type = torch_fake_tensor_translate(tensor)
    X = value_map[node.args[0].name][0]
    W = value_map[node.target + ".weight"][0]
    padding = module.padding
    attrs = {
        "dilation": DenseArrayBase.from_list(i64, module.dilation),
        "pad": DenseArrayBase.from_list(
            i64, (padding[0], padding[1], padding[0], padding[1])
        ),
        "group": IntegerAttr(module.groups, i64),
        "kernel_shape": DenseArrayBase.from_list(i64, module.kernel_size),
        "stride": DenseArrayBase.from_list(i64, module.stride),
    }
    if module.bias != None:
        return ConvBiasOp.build(
            operands=[X, W, value_map[node.target + ".bias"][0]],
            result_types=[result_type],
            attributes=attrs,
        )
    else:
        return ConvOp.build(
            operands=[X, W],
            result_types=[result_type],
            attributes=attrs,
        )


TORCH_OP_CONVERT_CALL = {torch.nn.modules.conv.Conv2d: torch_conv_convert}


def torch_module_translate(
    node: torch.fx.node.Node, value_map: dict, module: torch.nn.Module
) -> Operation:
    module_stack = node.meta["nn_module_stack"]
    target = node.target
    build_fn = TORCH_OP_CONVERT_CALL[module_stack[target][1]]
    return build_fn(node, value_map, module)
