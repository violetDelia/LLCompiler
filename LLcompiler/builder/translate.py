from xdsl.dialects.builtin import (
    IndexType,
    IntegerType,
    StringAttr,
    DenseIntOrFPElementsAttr,
    TensorType,
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
import onnx
import onnx.onnx_ml_pb2


def torch_module_translate(node: torch.fx.node.Node, value_map: dict):
    module_stack = node.meta["nn_module_stack"]
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



def onnx_node_translate(node:onnx.onnx_ml_pb2.NodeProto, op_map:dict,symbol_map: dict ):
    print(node)

def onnx_value_translate(
    value: onnx.onnx_ml_pb2.ValueInfoProto, symbol_map: dict  # not implemented yet
):
    ele_type = onnx_element_type_translate(value.type.tensor_type.elem_type)
    shape = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param == "":
            shape.append(dim.dim_value)
        else:
            if dim.dim_param.isdigit():
                shape.append(int(dim.dim_param))
            else:
                shape.append(DYNAMIC_INDEX)
    return TensorType(element_type=ele_type, shape=shape)


def onnx_weight_translate(weight: onnx.onnx_ml_pb2.TensorProto):
    weight_file = os.path.join(
        os.path.dirname(__file__),
        "LLcompiler_weight_temp",
        datetime.now().astimezone().isoformat(),
    )
    os.makedirs(os.path.dirname(weight_file),exist_ok = True)
    onnx._save_bytes(weight.raw_data, weight_file)
    ele_type = onnx_element_type_translate(weight.data_type)
    shape = [dim for dim in weight.dims]
    op = WeightOp.build(
        result_types=[TensorType(element_type=ele_type, shape=shape)],
        attributes={"weight_file": StringAttr(weight_file)},
    )
    return op


def onnx_element_type_translate(ele_type: int):
    match ele_type:
        case 1:
            return f32
        case 2:
            return IntegerType(8, Signedness.UNSIGNED)
        case 3:
            return IntegerType(8)
        case 4:
            return IntegerType(16, Signedness.UNSIGNED)
        case 5:
            return IntegerType(16)
        case 6:
            return IntegerType(32)
        case 7:
            return IntegerType(64)
        case 9:
            return IntegerType(1)
        case 10:
            return f16
        case 11:
            return f64
        case 12:
            return IntegerType(32, Signedness.UNSIGNED)
        case 13:
            return IntegerType(64, Signedness.UNSIGNED)
        case 16:
            return BFloat16Type
        case 21:
            return IntegerType(4, Signedness.UNSIGNED)
        case 22:
            return IntegerType(4)
        case _:
            raise NotImplementedError(f"Unsupported dtype: {ele_type}")
