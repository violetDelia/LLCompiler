from ...dialect.llh import MulOp, TorchSymbolicIntOp
from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_or_mate_tensor_translate,
    get_result_type,
    get_arg_value,
    commond_build_op,
    _expand_to_2_if_int,
    SPECIAL_RESULT_FAKE_INDEX_MAP,
    SPECIAL_GETITEM_IS_OPERAND_MAP,
)
import torch._ops as op
import torch.fx
import torch.nn.functional as F
from xdsl.ir import SSAValue, Operation, OpResult, Attribute, Mapping, Block
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
from torch._subclasses.fake_tensor import FakeTensor
from ...dialect.llh import TorchSymbolicIntOp, ConvBiasOp, ConvOp
from xdsl.irdl import IRDLOperation


@TORCH_FUNCTION_TRANSLATE("aten::convolution")
def convolution_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    block: Block,
):
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
    X = get_arg_value(node.args[0], value_map, block)
    W: OpResult = get_arg_value(node.args[1], value_map, block)
    padding = node.args[4]
    attrs = {
        "dilation": DenseArrayBase.from_list(i64, node.args[5]),
        "pad": DenseArrayBase.from_list(
            i64, (padding[0], padding[1], padding[0], padding[1])
        ),
        "group": IntegerAttr(node.args[8], i64),
        "kernel_shape": DenseArrayBase.from_list(i64, W.type.shape.data[-2:]),
        "stride": DenseArrayBase.from_list(i64, node.args[3]),
    }
    if node.args[6] != False:
        B = value_map[node.args[2].name][0]
        return ConvBiasOp.build(
            operands=[X, W, B],
            result_types=[result_type],
            attributes=attrs,
        )
    else:
        return ConvOp.build(
            operands=[X, W],
            result_types=[result_type],
            attributes=attrs,
        )
