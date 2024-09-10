from ...core.utility import Dict_Registry
from .fx_translate import TORCH_FUNCTION_TRANSLATE, torch_fake_tensor_translate
import torch._ops as op
import torch.fx
from xdsl.ir import SSAValue, Operation, OpResult
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
from ...dialect.llh import ConvBiasOp, ConvOp


@TORCH_FUNCTION_TRANSLATE("aten::sym_size.int")
def aten_sym_size_int_convert(node: torch.fx.node.Node, value_map: dict[str:[SSAValue]]):
    print(value_map)
    print(node)
    print(node.meta)
    

@TORCH_FUNCTION_TRANSLATE("aten::convolution")
def aten_convolution_convert(node: torch.fx.node.Node, value_map: dict[str:[SSAValue]]):
    print(node.meta)
    tensor = node.meta["val"]
    result_type = torch_fake_tensor_translate(tensor)
    X = value_map[node.args[0].name][0]
    W: OpResult = value_map[node.args[1].name][0]
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


def torch_function_translate(
    node: torch.fx.node.Node, value_map: dict[str, list[SSAValue]]
) -> Operation:
    target: op.OpOverload = node.target
    build_fn = TORCH_FUNCTION_TRANSLATE[target.name()]
    return build_fn(node, value_map)
