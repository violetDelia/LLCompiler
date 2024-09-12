from ...core.utility import Dict_Registry
from .fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_tensor_translate,
    get_result_type,
    get_arg_value,
)
import torch._ops as op
import torch.fx
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
)
from ...dialect.llh import ConvBiasOp, ConvOp, AddOp, DivOp, MulOp, DimOp, ReshapeOp
from ...dialect.llh_utility import build_llh_transpose
from torch._subclasses.fake_tensor import FakeTensor


def commond_build_op(
    op_build: callable,
    operand_nums: int,
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    block: Block,
    attrs: Mapping[str, Attribute | None] = None,
):
    out = get_result_type(node)
    if isinstance(out, FakeTensor):
        result_type = torch_fake_tensor_translate(out)
        return op_build(
            operands=[
                get_arg_value(node.args[n], value_map, block)
                for n in range(operand_nums)
            ],
            result_types=[result_type],
            attributes=attrs,
        )
    if isinstance(out, torch.SymInt):
        return op_build(
            operands=[
                get_arg_value(node.args[n], value_map, block)
                for n in range(operand_nums)
            ],
            result_types=[i64],
            attributes=attrs,
        )


@TORCH_FUNCTION_TRANSLATE("mul", "aten::mul.Tensor")
def builtin_mul_convert(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
):
    return commond_build_op(MulOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::add.Tensor", "add")
def builtin_add_convert(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
):
    return commond_build_op(AddOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("truediv", "aten::div.Tensor")
def builtin_truediv_convert(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
):
    return commond_build_op(DivOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::sym_size.int")
def aten_sym_size_int_convert(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
):
    return commond_build_op(DimOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::view")
def aten_view_tensor(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args[1])):
        dims.append(get_arg_value(node.args[1][dim], value_map, block))
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_FUNCTION_TRANSLATE("aten::t")
def aten_view_tensor(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
):
    input = get_arg_value(node.args[0], value_map, block)
    return build_llh_transpose(
        input, [x for x in reversed(range(input.type.get_num_dims()))]
    )


@TORCH_FUNCTION_TRANSLATE("aten::convolution")
def aten_convolution_convert(
    node: torch.fx.node.Node, value_map: dict[str:[SSAValue]], block: Block
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


def torch_function_translate(
    node: torch.fx.node.Node, value_map: dict[str, list[SSAValue]], block: Block
) -> Operation:
    target: op.OpOverload = node.target
    if type(target).__name__ == "builtin_function_or_method":
        build_fn = TORCH_FUNCTION_TRANSLATE[target.__name__]
    else:
        build_fn = TORCH_FUNCTION_TRANSLATE[target.name()]
    return build_fn(node, value_map, block)
