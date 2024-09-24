from .fx_translate import (
    TORCH_METHOD_TRANSLATE,
    torch_fake_tensor_translate,
    get_result_type,
    get_arg_value,
    torch_build_func,
    commond_build_op,
    _expand_to_2_if_int,
)
import torch
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
    DenseArrayBase,
    IntegerAttr,
)
import torch.fx
from xdsl.dialects.func import FuncOp, Call
from ...dialect.llh import (
    ConvBiasOp,
    ConvOp,
    MatmulOp,
    AddOp,
    TorchSymbolicIntOp,
    ReshapeOp,
    TransposeOp,
    ExpandOp,
)
from ...dialect.llh_utility import build_llh_transpose
from xdsl.ir import SSAValue, Operation, Block


@TORCH_METHOD_TRANSLATE("reshape")
def torch_reshape_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    print(node.args)
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args) - 1):
        dims.append(get_arg_value(node.args[dim + 1], value_map, block))
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_METHOD_TRANSLATE("expand")
def torch_reshape_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args) - 1):
        dims.append(get_arg_value(node.args[dim + 1], value_map, block))
    return ExpandOp(operands=[input, dims], result_types=[result_type])


@TORCH_METHOD_TRANSLATE("permute")
def torch_permute_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    perms = []
    for p in range(len(node.args) - 1):
        perms.append(p)
    return TransposeOp.build(
        operands=[input],
        attributes={"perms": DenseArrayBase.from_list(i64, perms)},
        result_types=[result_type],
    )

@TORCH_METHOD_TRANSLATE("view")
def aten_view_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args[1:])):
        dims.append(get_arg_value(node.args[1+dim], value_map, block))
    return ReshapeOp(operands=[input, dims], result_types=[result_type])