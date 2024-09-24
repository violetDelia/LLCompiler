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
    IntegerType,
    IntegerAttr,
    FloatAttr,
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
    DenseArrayBase,
)

from .llh import TorchSymbolicIntOp, SymbolicBindOp, ConstantOp, TransposeOp


def build_llh_scalar_tensor(val: int | float):
    if isinstance(val, int):
        type = TensorType([1], i64)
        value = DenseIntOrFPElementsAttr.create_dense_int(type, [val])
    if isinstance(val, float):
        type = TensorType([1], f32)
        value = DenseIntOrFPElementsAttr.create_dense_float(type, [val])
    return ConstantOp.build(attributes={"value": value}, result_types=[type])


def build_llh_constant(val: int | float):
    if isinstance(val, int):
        type = i64
        value = IntegerAttr(val, i64)
    if isinstance(val, float):
        type = f32
        value = FloatAttr(val, f32)
    return ConstantOp.build(attributes={"value": value}, result_types=[type])


def build_llh_transpose(input: SSAValue, perms: list[int]):
    tensor: TensorType = input.type
    shape = tensor.get_shape()
    new_shape = []
    for p in perms:
        new_shape.append(shape[p])
    result = TensorType(tensor.get_element_type(), new_shape)
    return TransposeOp.build(
        operands=[input],
        attributes={"perms": DenseArrayBase.from_list(i64, perms)},
        result_types=[result],
    )
