from .fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_tensor_translate,
    get_result_type,
    get_arg_value,
    commond_build_op,
    _expand_to_2_if_int,
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
)
from ...dialect.llh import (
    ConvBiasOp,
    EmptyOp,
    ConvOp,
    AddOp,
    DivOp,
    MulOp,
    DimOp,
    ReshapeOp,
    ConstantOp,
    TorchSymbolicIntOp,
    CatOp,
    FlattenOp,
    ReluOp,
    AdaptiveAvgPoolOp,
    MaxPoolOp,
    SubOp,
)
from ...dialect.llh_utility import build_llh_transpose, build_llh_constant
from torch._subclasses.fake_tensor import FakeTensor


@TORCH_FUNCTION_TRANSLATE("mul", "aten::mul.Tensor")
def builtin_mul_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(MulOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::add.Tensor", "add", "iadd")
def builtin_add_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(AddOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("sub")
def builtin_add_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(SubOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("truediv", "aten::div.Tensor", "floordiv")
def builtin_truediv_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(DivOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::sym_size.int")
def aten_sym_size_int_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(DimOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::relu", F.relu)
def aten_sym_size_int_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(ReluOp.build, 1, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("getitem")
def builtin_getitem_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    inputs = value_map[node.args[0].name]
    if (len(inputs) == 1) and isinstance(inputs[0].type, TensorType):
        if isinstance(node.args[1], slice):
            raise NotImplementedError("do not support slice current")
        dim: ConstantOp = build_llh_constant(node.args[1])
        block.add_op(dim)
        return DimOp(operands=[inputs[0], dim.result], result_types=[i64])

    else:
        raise NotImplementedError


@TORCH_FUNCTION_TRANSLATE("aten::view")
def aten_view_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    dims = []
    for dim in range(len(node.args[1])):
        dims.append(get_arg_value(node.args[1][dim], value_map, block))
    return ReshapeOp(operands=[input, dims], result_types=[result_type])


@TORCH_FUNCTION_TRANSLATE(F.max_pool2d)
def aten_view_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    arg_len = len(node.args)
    kernel_shape = node.args[1]
    stride = node.args[2] if (arg_len > 2) else [1, 1]
    padding = node.args[3] if (arg_len > 3) else [0, 0]
    dilation = node.args[4] if (arg_len > 4) else [1, 1]
    ceil_mode = node.args[5] if (arg_len > 5) else 0
    result_type = torch_fake_tensor_translate(get_result_type(node))
    input = get_arg_value(node.args[0], value_map, block)
    attrs = {
        "dilation": DenseArrayBase.from_list(i64, _expand_to_2_if_int(dilation)),
        "pad": DenseArrayBase.from_list(
            i64, (padding[0], padding[1], padding[0], padding[1])
        ),
        "kernel_shape": DenseArrayBase.from_list(
            i64, _expand_to_2_if_int(kernel_shape)
        ),
        "stride": DenseArrayBase.from_list(i64, _expand_to_2_if_int(stride)),
        "ceil_mode": BoolAttr(ceil_mode, i1),
    }
    return MaxPoolOp.build(
        operands=[input], attributes=attrs, result_types=[result_type]
    )


@TORCH_FUNCTION_TRANSLATE(F.adaptive_avg_pool2d)
def flatten_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(AdaptiveAvgPoolOp.build, 1, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("flatten")
def flatten_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    return commond_build_op(FlattenOp.build, 2, node, value_map, block)


@TORCH_FUNCTION_TRANSLATE("aten::t")
def aten_t_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    input = get_arg_value(node.args[0], value_map, block)
    return build_llh_transpose(
        input, [x for x in reversed(range(input.type.get_num_dims()))]
    )


@TORCH_FUNCTION_TRANSLATE("cat")
def cat_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    result_type = torch_fake_tensor_translate(get_result_type(node))
    operands = []
    for arg in node.args[0]:
        operands.append(get_arg_value(arg, value_map, block))
    attrs = {
        "dim": (
            IntegerAttr(node.kwargs["dim"], i64)
            if "dim" in node.kwargs
            else IntegerAttr(node.args[1], i64)
        )
    }
    return CatOp(operands=[operands], attributes=attrs, result_types=[result_type])


@TORCH_FUNCTION_TRANSLATE("aten::convolution")
def aten_convolution_convert(
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


@TORCH_FUNCTION_TRANSLATE("empty")
def builtin_mul_convert(
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


def torch_function_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
) -> Operation:
    target: op.OpOverload = node.target
    if type(target).__name__ == "builtin_function_or_method":
        build_fn = TORCH_FUNCTION_TRANSLATE[target.__name__]
    else:
        build_fn = TORCH_FUNCTION_TRANSLATE[target.name()]
    return build_fn(node, value_map, symbol_map, block)
