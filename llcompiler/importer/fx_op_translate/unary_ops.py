from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    TORCH_METHOD_TRANSLATE,
    TORCH_MODULE_TRANSLATE,
    torch_fake_or_mate_tensor_translate,
    get_result_type,
    get_arg_value,
    commond_build_op,
    torch_dtype_translate,
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
from ...dialect.llh import TorchSymbolicIntOp, AbsOp, ReluOp, SqrtOp, RsqrtOp, DivOp


@TORCH_FUNCTION_TRANSLATE("aten::rsqrt", "prims::rsqrt")
def rsqrt_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(RsqrtOp.build, 1, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE(torch._sym_sqrt, "aten::sqrt", "prims::sqrt")
def sqrt_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(SqrtOp.build, 1, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::relu", F.relu)
def relu_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(ReluOp.build, 1, node, value_map, block)


@TORCH_MODULE_TRANSLATE(torch.nn.modules.activation.ReLU)
def torch_relu_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.activation.ReLU,
    block: Block,
):
    return commond_build_op(ReluOp.build, 1, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("abs", "aten::abs")
def abs_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(AbsOp.build, 1, node, value_map, block)


@TORCH_METHOD_TRANSLATE("abs")
def abs_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(AbsOp.build, 1, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::reciprocal")
def reciprocal_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    out = get_result_type(node)
    result_type = torch_fake_or_mate_tensor_translate(out)
    input = get_arg_value(node.args[0], value_map, block)
    one = get_arg_value(
        1,
        value_map,
        block,
        const_tensor=True,
        const_type=torch_dtype_translate(out.dtype),
    )
    return DivOp(
        operands=[one, input],
        result_types=[result_type],
    )
