from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    TORCH_MODULE_TRANSLATE,
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
from ...dialect.llh import DimOp, TorchSymbolicIntOp, LayerNormOp


@TORCH_MODULE_TRANSLATE(torch.nn.modules.normalization.LayerNorm)
def torch_layernorm_convert(
    node: torch.fx.node.Node,
    value_map: dict,
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.modules.normalization.LayerNorm,
    block: Block,
):
    result_type = torch_fake_or_mate_tensor_translate(get_result_type(node))
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
