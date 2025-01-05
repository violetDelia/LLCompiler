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
from ...dialect.llh import (
    TorchSymbolicIntOp,
    BatchNormOp,
    ModeAttr,
    ModeEnum,
    BatchNormInferenceOp,
)
from xdsl.irdl import IRDLOperation


@TORCH_FUNCTION_TRANSLATE("aten::_native_batch_norm_legit_no_training")
def batch_norm_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    block: Block,
):
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    weight = get_arg_value(node.args[1], value_map, block)
    bias = get_arg_value(node.args[2], value_map, block)
    input_mean = get_arg_value(node.args[3], value_map, block)
    input_var = get_arg_value(node.args[4], value_map, block)
    attrs = {
        "epsilon": FloatAttr(node.args[6], f64),
        "momentum": FloatAttr(node.args[5], f64),
        "feature_index": IntegerAttr(1, i64),
        "mode": ModeAttr([ModeEnum.Inference]),
    }
    return BatchNormInferenceOp.build(
        operands=[input, weight, bias, input_mean, input_var],
        attributes=attrs,
        result_types=[result_type],
    )


@TORCH_FUNCTION_TRANSLATE("aten::_native_batch_norm_legit_functional")
def batch_norm_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    block: Block,
):
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    weight = get_arg_value(node.args[1], value_map, block)
    bias = get_arg_value(node.args[2], value_map, block)
    input_mean = get_arg_value(node.args[3], value_map, block)
    input_var = get_arg_value(node.args[4], value_map, block)
    attrs = {
        "epsilon": FloatAttr(node.args[6], f64),
        "momentum": FloatAttr(node.args[7], f64),
        "feature_index": IntegerAttr(1, i64),
        "mode": ModeAttr([ModeEnum.Inference]),
    }
    op = BatchNormOp.build(
        operands=[input, weight, bias, input_mean, input_var],
        attributes=attrs,
        result_types=[result_type],
    )
    return op
