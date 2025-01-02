from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_or_mate_tensor_translate,
    get_result_type,
    get_arg_value,
    commond_build_op,
    _expand_to_2_if_int,
    _updata_torch_symbol_bind,
    SPECIAL_RESULT_FAKE_INDEX_MAP,
    SPECIAL_GETITEM_IS_OPERAND_MAP,
)
from xdsl.dialects.builtin import (
    TensorType,
    IntegerType,
    i64,
    i32,
    i16,
    i1,
    f16,
    f32,
    f64,
    DYNAMIC_INDEX,
    DenseArrayBase,
    IntegerAttr,
    BoolAttr,
    DenseIntOrFPElementsAttr,
    FloatAttr,
)
from ...dialect.llh_utility import build_llh_transpose, build_llh_constant
import torch._ops as op
import torch.fx
import torch.nn.functional as F
from xdsl.ir import SSAValue, Operation, OpResult, Attribute, Mapping, Block
from torch._subclasses.fake_tensor import FakeTensor
from ...dialect.llh import TorchSymbolicIntOp, CompareOp, CompareAttr, CompareEnum


def get_compare_type(node: torch.fx.node.Node, value_map: dict[str:[SSAValue]]):
    lhs = node.args[0]
    rhs = node.args[1]
    if (isinstance(lhs, (int, float))) and isinstance(rhs, (int, float)):
        return i64 if isinstance(lhs, int) else f32
    if isinstance(rhs, (int, float)):
        return value_map[lhs.name][0].type
    if isinstance(lhs, (int, float)):
        return value_map[rhs.name][0].type


@TORCH_FUNCTION_TRANSLATE("prims::eq", "aten::eq.Scalar")
def eq_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
    type: TensorType = get_compare_type(node, value_map)
    if isinstance(type, TensorType):
        type = type.element_type
    lhs = get_arg_value(
        node.args[0], value_map, block, const_tensor=True, const_type=type
    )
    rhs = get_arg_value(
        node.args[1], value_map, block, const_tensor=True, const_type=type
    )
    attrs = {"kind": CompareAttr([CompareEnum.EQ])}
    return CompareOp(operands=[lhs, rhs], result_types=[result_type], attributes=attrs)


@TORCH_FUNCTION_TRANSLATE("prims::le", "aten::le.Scalar")
def le_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
    type: TensorType = get_compare_type(node, value_map)
    if isinstance(type, TensorType):
        type = type.element_type
    lhs = get_arg_value(
        node.args[0], value_map, block, const_tensor=True, const_type=type
    )
    rhs = get_arg_value(
        node.args[1], value_map, block, const_tensor=True, const_type=type
    )
    attrs = {"kind": CompareAttr([CompareEnum.LE])}
    return CompareOp(operands=[lhs, rhs], result_types=[result_type], attributes=attrs)
