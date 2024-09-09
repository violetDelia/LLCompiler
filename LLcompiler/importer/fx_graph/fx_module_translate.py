from ...core.utility import Dict_Registry
from .fx_translate import *

module_translate = Dict_Registry()


@module_translate(torch.nn.modules.conv.Conv2d)
def torch_conv_convert(
    node: torch.fx.node.Node, value_map: dict, module: torch.nn.modules.conv._ConvNd
):
    tensor = node.meta["example_value"]
    result_type = torch_fake_tensor_translate(tensor)
    X = value_map[node.args[0].name][0]
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
        return ConvBiasOp.build(
            operands=[X, W, value_map[node.target + ".bias"][0]],
            result_types=[result_type],
            attributes=attrs,
        )
    else:
        return ConvOp.build(
            operands=[X, W],
            result_types=[result_type],
            attributes=attrs,
        )


def torch_module_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    module: torch.nn.Module,
) -> Operation:
    module_stack = node.meta["nn_module_stack"]
    target = node.target
    build_fn = module_translate[module_stack[target][1]]
    return build_fn(node, value_map, module)
