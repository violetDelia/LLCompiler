from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    torch_fake_or_mate_tensor_translate,
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
from ...dialect.llh import DimOp, TorchSymbolicIntOp, ExtractOp, ConstantOp


@TORCH_FUNCTION_TRANSLATE("getitem")
def builtin_getitem_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
):
    target = node.args[0].target
    if target in SPECIAL_RESULT_FAKE_INDEX_MAP:
        if SPECIAL_RESULT_FAKE_INDEX_MAP[target] == node.args[1]:
            value_map[node.name] = value_map[node.args[0].name]
            return None
        else:
            value_map[node.name] = None
            return None
    inputs = value_map[node.args[0].name]
    if (len(inputs) == 1) and isinstance(inputs[0].type, TensorType):
        out = get_result_type(node)
        # Slice
        if len(node.args) > 1 and isinstance(node.args[1], slice):
            print(node)
            print(node.args)
            raise NotImplementedError("do not support slice current")
        # Dim
        elif isinstance(out, torch.SymInt):
            dim: ConstantOp = build_llh_constant(node.args[1])
            block.add_op(dim)
            return DimOp(operands=[inputs[0], dim.result], result_types=[i64])
        elif isinstance(out, FakeTensor):
            index: ConstantOp = build_llh_constant(node.args[1])
            block.add_op(index)
            extract_op = ExtractOp(
                operands=[inputs[0], index.result],
                result_types=[torch_fake_or_mate_tensor_translate(out)],
            )
            return extract_op
        else:
            raise ValueError(node, type(out))

    else:
        value_map[node.name] = [value_map[node.args[0].name][node.args[1]]]
