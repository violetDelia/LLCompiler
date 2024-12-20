from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_or_mate_tensor_translate,
    torch_symbol_translate,
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
from ...dialect.llh import TorchSymbolicIntOp, BroadCastToOp


@TORCH_FUNCTION_TRANSLATE("prims::broadcast_in_dim")
def broadcast_in_dim_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    res_tensor = get_result_type(node)
    result_type = torch_fake_or_mate_tensor_translate(res_tensor)
    input = get_arg_value(node.args[0], value_map, block)
    cast_dims = []
    res_dims = node.args[1]
    for dim in res_dims:
        if isinstance(dim, int):
            const_dim = build_llh_constant(dim)
            block.add_op(const_dim)
            cast_dims.append(const_dim.result)
        else:
            dim = value_map[dim.name][0]
            cast_dims.append(dim)
    attrs = {"cast_dims": DenseArrayBase.from_list(i64, node.args[2])}
    op = BroadCastToOp(
        operands=[input, cast_dims], attributes=attrs, result_types=[result_type]
    )
    return op
