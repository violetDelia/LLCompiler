import sympy.core.core
import sympy.core.facts
import sympy.core.mul
from sympy.core.numbers import Integer
import sympy.core.numbers
from sympy.core.symbol import Symbol
import sympy.core
import torch.fx.experimental
import torch.fx.experimental.sym_node
from xdsl.ir import SSAValue, Block
from xdsl.ir.affine.affine_expr import (
    AffineSymExpr,
    AffineConstantExpr,
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
)
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
    DenseIntOrFPElementsAttr,
)
from ...dialect.llh_utility import build_llh_constant
from ...core.utility import Dict_Registry
from datetime import datetime
import torch.nn
from torch._subclasses.fake_tensor import FakeTensor
from ...core.utility import run_time
import os
import numpy as np
import torch.fx
from torch.fx.experimental.sym_node import SymNode
from ...dialect.llh import SymbolicIntOp, SymbolicShapeBindOp, ConstantOp
import sympy


def torch_symbol_translate(symbol: torch.SymInt, symbol_map: dict[str, SymbolicIntOp]):
    name: str = symbol.node.expr.name
    atts = {"value": StringAttr(name)}
    op = SymbolicIntOp(attributes=atts, result_types=[i64])
    symbol_map[name] = op
    return op


def _generate_affine_symbolic(
    arg,
    symbol_map: dict[str, SymbolicIntOp],
    affine_expr_map: dict[str, AffineSymExpr],
    bind_symbols: list[SSAValue],
):
    if isinstance(arg, sympy.core.Number):
        return AffineConstantExpr(int(arg))
    elif isinstance(arg, sympy.core.Add):
        return AffineBinaryOpExpr(
            AffineBinaryOpKind.Add,
            _generate_affine_symbolic(
                arg.args[1], symbol_map, affine_expr_map, bind_symbols
            ),
            _generate_affine_symbolic(
                arg.args[0], symbol_map, affine_expr_map, bind_symbols
            ),
        )
    elif isinstance(arg, sympy.core.mul.Mul):

        return AffineBinaryOpExpr(
            AffineBinaryOpKind.Mul,
            _generate_affine_symbolic(
                arg.args[1], symbol_map, affine_expr_map, bind_symbols
            ),
            _generate_affine_symbolic(
                arg.args[0], symbol_map, affine_expr_map, bind_symbols
            ),
        )
    elif type(arg).__name__ == "FloorDiv":
        return AffineBinaryOpExpr(
            AffineBinaryOpKind.FloorDiv,
            _generate_affine_symbolic(
                arg.args[0], symbol_map, affine_expr_map, bind_symbols
            ),
            _generate_affine_symbolic(
                arg.args[1], symbol_map, affine_expr_map, bind_symbols
            ),
        )
    elif isinstance(arg, Symbol):
        name: str = arg.name
        if name in symbol_map:
            if name not in affine_expr_map:
                affine_expr = AffineSymExpr(len(bind_symbols))
                affine_expr_map[name] = affine_expr
                bind_symbols.append(symbol_map[name].result)
            return affine_expr_map[name]
        else:
            raise NotImplementedError("name must in symbol map")
    else:
        raise NotImplementedError(arg, type(arg).__name__)


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
            affine_exp = _generate_affine_symbolic(
                dim.node.expr, symbol_map, affine_expr_map, bind_symbols
            )
            results.append(affine_exp)
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


def get_result_type(
    node: torch.fx.node.Node,
):
    if "val" in node.meta:
        return node.meta["val"]
    if "example_value" in node.meta:
        return node.meta["example_value"]
    raise ValueError("No example_value found in node meta")


def get_arg_value(
    arg: str | int | float,
    value_map: dict[str:[SSAValue]],
    block: Block,
    index: int = 0,
):
    if isinstance(arg, torch.fx.node.Node):
        return value_map[arg.name][index]
    elif isinstance(arg, int) or isinstance(arg, float):
        const = build_llh_constant(arg)
        block.add_op(const)
        return const.result
    else:
        raise NotImplementedError(arg, type(arg))


TORCH_FUNCTION_TRANSLATE = Dict_Registry()

TORCH_MODULE_TRANSLATE = Dict_Registry()
