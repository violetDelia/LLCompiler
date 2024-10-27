from .fx_translate import (
    TORCH_MODULE_TRANSLATE,
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
    FloatAttr,
    Float64Type,
    IntegerAttr,
    BoolAttr,
)
import torch.fx
from xdsl.dialects.func import FuncOp, Call
from ...dialect.llh import (
    ConvBiasOp,
    ConvOp,
    MatmulOp,
    AddOp,
    TorchSymbolicIntOp,
    DropOp,
    LayerNormOp,
    ReluOp,
    MaxPoolOp,
    AdaptiveAvgPoolOp,
    BatchNormOp,
    FlattenOp,
)
from torch._subclasses.fake_tensor import FakeTensor

from ...dialect.llh_utility import build_llh_transpose
from xdsl.ir import SSAValue, Operation, OpResult, Attribute, Mapping, Block

# define


@TORCH_MODULE_TRANSLATE(torch.nn.modules.transformer.Transformer)
def torch_Transformer_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.transformer.Transformer,
    block: Block,
):
    raise NotImplementedError


@TORCH_MODULE_TRANSLATE(torch.nn.modules.conv.Conv2d)
def torch_conv_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.conv._ConvNd,
    block: Block,
):
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


@TORCH_MODULE_TRANSLATE(torch.nn.modules.linear.Linear)
def torch_linear_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.conv._ConvNd,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    lhs = get_arg_value(node.args[0], value_map, block)
    weight = value_map[node.target + ".weight"][0]
    rhs = build_llh_transpose(
        weight,
        [x for x in reversed(range(weight.type.get_num_dims()))],
    )
    block.add_op(rhs)
    matmul = MatmulOp.build(operands=[lhs, rhs], result_types=[result_type])
    if module.bias == None:
        return matmul
    block.add_op(matmul)
    bias = value_map[node.target + ".bias"][0]
    return AddOp.build(operands=[matmul.result, bias], result_types=[result_type])


@TORCH_MODULE_TRANSLATE(torch.nn.modules.dropout.Dropout)
def torch_drop_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.dropout.Dropout,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    attrs = {"p": FloatAttr(module.p, f64)}
    return DropOp.build(operands=[input], attributes=attrs, result_types=[result_type])


@TORCH_MODULE_TRANSLATE(torch.nn.modules.normalization.LayerNorm)
def torch_layernorm_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.normalization.LayerNorm,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    weight = value_map[node.target + ".weight"][0]
    bias = value_map[node.target + ".bias"][0]
    attrs = {
        "epsilon": FloatAttr(module.eps, f64),
        "axis": IntegerAttr(result_type.get_num_dims() - 1, i64),
    }
    return LayerNormOp(
        operands=[input, weight, bias], attributes=attrs, result_types=[result_type]
    )


@TORCH_MODULE_TRANSLATE(torch.nn.modules.activation.MultiheadAttention)
def torch_multiheadattention_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.activation.MultiheadAttention,
    block: Block,
):
    raise NotImplementedError


@TORCH_MODULE_TRANSLATE(torch.nn.modules.activation.ReLU)
def torch_relu_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.activation.ReLU,
    block: Block,
):
    return commond_build_op(ReluOp.build, 1, node, value_map, block)


@TORCH_MODULE_TRANSLATE(torch.nn.modules.flatten.Flatten)
def torch_relu_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.flatten.Flatten,
    block: Block,
):
    if module.end_dim != -1:
        raise ValueError("改成reshape")
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dim = get_arg_value(module.start_dim, value_map, block)
    return FlattenOp.build(operands=[input, dim], result_types=[result_type])


@TORCH_MODULE_TRANSLATE(torch.nn.modules.pooling.MaxPool2d)
def torch_maxpool_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.pooling.MaxPool2d,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    padding = _expand_to_2_if_int(module.padding)
    attrs = {
        "dilation": DenseArrayBase.from_list(i64, _expand_to_2_if_int(module.dilation)),
        "pad": DenseArrayBase.from_list(
            i64, (padding[0], padding[1], padding[0], padding[1])
        ),
        "kernel_shape": DenseArrayBase.from_list(
            i64, _expand_to_2_if_int(module.kernel_size)
        ),
        "stride": DenseArrayBase.from_list(i64, _expand_to_2_if_int(module.stride)),
        "ceil_mode": BoolAttr(module.ceil_mode, i1),
    }
    return MaxPoolOp.build(
        operands=[input], attributes=attrs, result_types=[result_type]
    )


@TORCH_MODULE_TRANSLATE(torch.nn.modules.pooling.AdaptiveAvgPool2d)
def torch_adaptive_avgpool_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.pooling.AdaptiveAvgPool2d,
    block: Block,
):
    attrs = {"out_size": DenseArrayBase.from_list(i64, module.output_size)}
    return commond_build_op(AdaptiveAvgPoolOp.build, 1, node, value_map, block, attrs)


@TORCH_MODULE_TRANSLATE(
    torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.BatchNorm1d
)
def torch_adaptive_avgpool_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.batchnorm.BatchNorm2d,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    weight = value_map[node.target + ".weight"][0]
    bias = value_map[node.target + ".bias"][0]
    input_mean = value_map[node.target + ".running_mean"][0]
    input_var = value_map[node.target + ".running_var"][0]
    attrs = {
        "epsilon": FloatAttr(module.eps, f64),
        "momentum": FloatAttr(module.momentum, f64),
        "feature_index": IntegerAttr(1, i64),
    }
    return BatchNormOp.build(
        operands=[input, weight, bias, input_mean, input_var],
        attributes=attrs,
        result_types=[result_type],
    )
