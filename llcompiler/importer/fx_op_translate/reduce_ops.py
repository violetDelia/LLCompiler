from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_or_mate_tensor_translate,
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
from ...dialect.llh_utility import (
    build_llh_transpose,
    build_llh_constant,
    build_value_dims,
)
import torch._ops as op
import torch.fx
import torch.nn.functional as F
from xdsl.ir import SSAValue, Operation, OpResult, Attribute, Mapping, Block
from torch._subclasses.fake_tensor import FakeTensor
from ...dialect.llh import MaxPoolOp, TorchSymbolicIntOp, ReduceMaxOp, ReshapeOp


@TORCH_FUNCTION_TRANSLATE("aten::amax")
def amax_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    input: SSAValue = get_arg_value(node.args[0], value_map, block)
    input_type: TensorType = input.type
    rank = input_type.get_shape().__len__()
    axis = node.args[1]
    axis = [dim if dim > 0 else dim + rank for dim in axis]
    keep_dim = node.args[2] if len(node.args) > 2 else False
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
    attrs = {"axis": DenseArrayBase.from_list(i64, axis)}
    if not keep_dim:
        return ReduceMaxOp.build(
            operands=[input], attributes=attrs, result_types=[result_type]
        )
    else:
        input_type: TensorType = input.type
        shape = [dim for dim in input_type.get_shape()]
        reduce_out_shape = [dim for i, dim in enumerate(shape) if i not in axis]
        reduce_out_type = TensorType(input_type.element_type, reduce_out_shape)
        reduce = ReduceMaxOp.build(
            operands=[input], attributes=attrs, result_types=[reduce_out_type]
        )
        block.add_op(reduce)
        dims: list = build_value_dims(input, block)
        one = build_llh_constant(1)
        block.add_op(one)
        for dim in axis:
            dims[dim] = one
        return ReshapeOp(operands=[reduce.result, dims], result_types=[result_type])
