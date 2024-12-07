from ...dialect.llh import MulOp, TorchSymbolicIntOp
from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    TORCH_MODULE_TRANSLATE,
    torch_fake_tensor_translate,
    get_result_type,
    get_arg_value,
    commond_build_op,
    _expand_to_2_if_int,
    _updata_torch_symbol_bind,
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


@TORCH_FUNCTION_TRANSLATE("aten::convolution")
def convolution_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
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


@TORCH_MODULE_TRANSLATE(torch.nn.modules.conv.Conv2d)
def torch_conv_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.conv._ConvNd,
    block: Block,
):
    print(get_result_type(node))
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    weight = value_map[node.target + ".weight"][0]
    padding = _expand_to_2_if_int(module.padding)
    attrs = {
        "dilation": DenseArrayBase.from_list(i64, _expand_to_2_if_int(module.dilation)),
        "pad": DenseArrayBase.from_list(
            i64, (padding[0], padding[1], padding[0], padding[1])
        ),
        "group": IntegerAttr(module.groups, i64),
        "kernel_shape": DenseArrayBase.from_list(
            i64, _expand_to_2_if_int(module.kernel_size)
        ),
        "stride": DenseArrayBase.from_list(i64, _expand_to_2_if_int(module.stride)),
    }
    if module.bias != None:
        bias = value_map[node.target + ".bias"][0]
        return ConvBiasOp.build(
            operands=[input, weight, bias],
            result_types=[result_type],
            attributes=attrs,
        )
    return ConvOp.build(
        operands=[input, weight],
        result_types=[result_type],
        attributes=attrs,
    )
