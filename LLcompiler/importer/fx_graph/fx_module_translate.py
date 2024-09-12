from .fx_translate import (
    TORCH_MODULE_TRANSLATE,
    torch_fake_tensor_translate,
    get_result_type,
    get_arg_value,
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
from ...dialect.llh import ConvBiasOp, ConvOp, MatmulOp, AddOp
from ...dialect.llh_utility import build_llh_transpose
from xdsl.ir import SSAValue, Operation, Block


@TORCH_MODULE_TRANSLATE(torch.nn.modules.conv.Conv2d)
def torch_conv_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    module: torch.nn.modules.conv._ConvNd,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_fake_tensor(node))
    X = get_arg_value(node.args[0], value_map, block)
    W = value_map[node.target + ".weight"][0]
    padding = module.padding
    attrs = {
        "dilation": DenseArrayBase.from_list(i64, module.dilation),
        "pad": DenseArrayBase.from_list(
            i64, (padding[0], padding[1], padding[0], padding[1])
        ),
        "group": IntegerAttr(module.groups, i64),
        "kernel_shape": DenseArrayBase.from_list(i64, module.kernel_size),
        "stride": DenseArrayBase.from_list(i64, module.stride),
    }
    if module.bias != None:
        B = value_map[node.target + ".bias"][0]
        return ConvBiasOp.build(
            operands=[X, W, B],
            result_types=[result_type],
            attributes=attrs,
        )
    return ConvOp.build(
        operands=[X, W],
        result_types=[result_type],
        attributes=attrs,
    )


@TORCH_MODULE_TRANSLATE(torch.nn.modules.linear.Linear)
def torch_conv_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    module: torch.nn.modules.conv._ConvNd,
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_fake_tensor(node))
    lhs = get_arg_value(node.args[0], value_map, block)
    weight = value_map[node.target + ".weight"][0]
    rhs = build_llh_transpose(
        weight,
        [x for x in reversed(range(weight.type.get_num_dims()))],
    )
    block.add_op(rhs)
    matmul = MatmulOp.build(operands=[lhs, rhs], result_types=[result_type])
    if module.bias != None:
        return matmul
    bais = value_map[node.target + ".bias"][0]
    return AddOp.build(operands=[matmul.result, bais], result_types=[result_type])


def torch_module_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    module: torch.nn.Module,
    block: Block,
) -> Operation:
    module_stack = node.meta["nn_module_stack"]
    target = node.target
    build_fn = TORCH_MODULE_TRANSLATE[module_stack[target][1]]
    return build_fn(node, value_map, module, block)
