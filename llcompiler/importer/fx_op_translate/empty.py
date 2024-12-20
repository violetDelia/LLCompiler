from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_tensor_translate,
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
from ...dialect.llh import DimOp, TorchSymbolicIntOp, EmptyOp


@TORCH_FUNCTION_TRANSLATE("aten::empty.memory_format")
def empty_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    dims = get_arg_value(node.args[0], value_map, block)
    op = EmptyOp.build(operands=[dims], result_types=[result_type])
    return op


@TORCH_FUNCTION_TRANSLATE("empty")
def empty_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    dims = [
        get_arg_value(node.args[i], value_map, block) for i in range(len(node.args))
    ]
    op = EmptyOp.build(operands=[dims], result_types=[result_type])
    return op
