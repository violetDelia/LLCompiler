from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
import torch.fx.experimental
import torch.fx.experimental.sym_node
from xdsl.ir import SSAValue
from xdsl.ir.affine.affine_expr import AffineSymExpr, AffineConstantExpr
from xdsl.ir.affine.affine_map import AffineMap
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


def _generate_symbol_expression(
    dim: torch.SymInt,
    symbol_map: dict[str, SymbolicIntOp],
    affine_expr_map: dict[str, AffineSymExpr],
    bind_symbols: list[SSAValue],
    results: list[AffineSymExpr],
):
    node: SymNode = dim.node
    exp: Symbol = node.expr
    name: str = exp.name
    if isinstance(exp, Symbol):
        if name in symbol_map:
            if name not in affine_expr_map:
                affine_expr_map[name] = AffineSymExpr(len(bind_symbols))
                bind_symbols.append(symbol_map[name].result)
            results.append(affine_expr_map[name])
        else:
            raise NotImplementedError
    else:
        pass


def torch_bind_shape(
    operand: SSAValue, tensor: FakeTensor, symbol_map: dict[str, SymbolicIntOp]
):
    bind_symbols: list[SSAValue] = []
    affine_expr_map: dict[str, AffineSymExpr] = dict()
    results: list[AffineSymExpr] = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            results.append(AffineConstantExpr(dim))
            continue
        elif str(dim).isdigit():
            results.append(AffineConstantExpr(int(dim)))
            continue
        else:
            _generate_symbol_expression(
                dim, symbol_map, affine_expr_map, bind_symbols, results
            )
    map = AffineMap(0, len(symbol_map), results=results)
    expressions = AffineMapAttr(map)
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
