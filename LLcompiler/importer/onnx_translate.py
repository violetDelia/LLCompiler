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


def onnx_node_translate(
    node: onnx.onnx_ml_pb2.NodeProto, op_map: dict, symbol_map: dict
):
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
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)
    onnx._save_bytes(weight.raw_data, weight_file)
    ele_type = onnx_element_type_translate(weight.data_type)
    shape = [dim for dim in weight.dims]
    op = WeightOp.build(
        result_types=[TensorType(element_type=ele_type, shape=shape)],
        attributes={"weight_file": StringAttr(weight_file)},
    )
    return op


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

ONNX_ELEMENT_TYPE_TO_MLIR_TYPE = {
    1: f32,
    2: IntegerType(8, Signedness.UNSIGNED),
    3: IntegerType(8),
    4: IntegerType(16, Signedness.UNSIGNED),
    5: IntegerType(16),
    6: IntegerType(32),
    7: IntegerType(64),
    9: IntegerType(1),
    10: f16,
    11: f64,
    12: IntegerType(32, Signedness.UNSIGNED),
    13: IntegerType(64, Signedness.UNSIGNED),
    16: BFloat16Type,
    21: IntegerType(4, Signedness.UNSIGNED),
    22: IntegerType(4),
}


def onnx_element_type_translate(ele_type: int):
    return ONNX_ELEMENT_TYPE_TO_MLIR_TYPE[ele_type]
