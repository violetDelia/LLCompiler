from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_tensor_translate,
    TORCH_MODULE_TRANSLATE,
    get_result_type,
    get_arg_value,
    commond_build_op,
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
from ...dialect.llh import MaxPoolOp,TorchSymbolicIntOp


@TORCH_FUNCTION_TRANSLATE(F.max_pool2d, "aten::max_pool2d_with_indices")
def max_pool2d_convert(
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