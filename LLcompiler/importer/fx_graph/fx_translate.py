from sympy.core.numbers import Integer
import torch.fx.experimental
import torch.fx.experimental.sym_node
from xdsl.ir import SSAValue
from xdsl.ir.affine.affine_expr import AffineSymExpr
from xdsl.dialects.builtin import (
    TensorType,
    i64,
    i32,
    i16,
    i1,
    f16,
    f32,
    f64,
    DYNAMIC_INDEX,
    StringAttr,
    i64,
    SymbolRefAttr,
    AffineMapAttr,
)
from ...core.utility import Dict_Registry
from datetime import datetime
import torch.nn
from torch._subclasses.fake_tensor import FakeTensor
from ...core.utility import run_time
import os
import numpy as np
import torch.fx
from torch.fx.experimental.sym_node import SymNode
from ...dialect.llh import SymbolicIntOp, SymbolicShapeBindOp
import sympy


def torch_symbol_translate(symbol: torch.SymInt, symbol_map: dict[str, SymbolicIntOp]):
    name: str = symbol.node.expr.name
    atts = {"value": StringAttr(name)}
    op = SymbolicIntOp(attributes=atts, result_types=[i64])
    symbol_map[name] = op
    return op


def torch_bind_shape(
    operand: SSAValue, tensor: FakeTensor, symbol_map: dict[str, SymbolicIntOp]
):
    bind_symbols = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            continue
        elif str(dim).isdigit():
            continue
        else:
            node: SymNode = dim.node
            print(dim)
    expressions = AffineMapAttr(AffineSymExpr(0))
    return SymbolicShapeBindOp(
        operands=[operand, bind_symbols], attributes={"expressions": expressions}
    )


def torch_fake_tensor_translate(tensor: FakeTensor):
    ele_type = torch_dtype_translate(tensor.dtype)
    shape = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            shape.append(dim)
        if isinstance(dim, torch.SymInt):
            if str(dim).isdigit():
                shape.append(dim.node.int_())
            else:
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


TORCH_FUNCTION_TRANSLATE = Dict_Registry()

TORCH_MODULE_TRANSLATE = Dict_Registry()
