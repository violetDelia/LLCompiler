from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    TORCH_METHOD_TRANSLATE,
    torch_fake_tensor_translate,
    get_result_type,
    get_arg_value,
    torch_symbol_translate,
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
from ...dialect.llh_utility import (
    build_llh_transpose,
    build_llh_constant,
    build_value_dims,
)
import torch._ops as op
import torch.fx
import torch.nn.functional as F
from xdsl.ir import SSAValue, Operation, OpResult, Attribute, Mapping, Block
from torch._subclasses.fake_tensor import FakeTensor
from ...dialect.llh import TorchSymbolicIntOp, ReshapeOp,DivOp


@TORCH_FUNCTION_TRANSLATE("aten::view")
def aten_view_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args[1])):
        dims.append(get_arg_value(node.args[1][dim], value_map, block))
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_METHOD_TRANSLATE("reshape")
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
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_METHOD_TRANSLATE("view")
def view_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args[1:])):
        dims.append(get_arg_value(node.args[1 + dim], value_map, block))
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_FUNCTION_TRANSLATE("aten::unsqueeze")
def unsqueeze_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    res_tensor: FakeTensor = get_result_type(node)
    result_type = torch_fake_tensor_translate(res_tensor)
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in res_tensor.shape:
        if isinstance(dim, int):
            const = build_llh_constant(dim)
            block.add_op(const)
            dims.append(const)
        elif isinstance(dim, torch.SymInt):
            symbol = torch_symbol_translate(dim, symbol_map)
            block.add_op(symbol)
            dims.append(symbol)
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_METHOD_TRANSLATE("unsqueeze")
def unsqueeze_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    res_tensor: FakeTensor = get_result_type(node)
    result_type = torch_fake_tensor_translate(res_tensor)
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in res_tensor.shape:
        if isinstance(dim, int):
            const = build_llh_constant(dim)
            block.add_op(const)
            dims.append(const)
        elif isinstance(dim, torch.SymInt):
            symbol = torch_symbol_translate(dim, symbol_map)
            block.add_op(symbol)
            dims.append(symbol)
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_FUNCTION_TRANSLATE("prims::collapse_view")
def collapse_view_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    input = get_arg_value(node.args[0], value_map, block)
    res_tensor: FakeTensor = get_result_type(node)
    result_type: TensorType = torch_fake_tensor_translate(res_tensor)
    dims = build_value_dims(input, block)
    assert len(node.args[1:]) == 2
    new_dim = build_llh_constant(1)
    for i in node.args[1:]:
        new_dim = DivOp(operands=[new_dim, dims[i]], result_types=[i64])
        block.add_op(new_dim)
        dims[i]= None
    dims[node.args[-1]] = new_dim
    new_dims = [dim for dim in dims if dim is not None]
    return ReshapeOp(operands=[input, new_dims], result_types=[result_type])
