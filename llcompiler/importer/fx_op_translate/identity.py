from ..fx_translate import (
    TORCH_FUNCTION_TRANSLATE,
    
    torch_fake_or_mate_tensor_translate,
    get_result_type,
    get_arg_value,
    commond_build_op,
    _expand_to_2_if_int,
    
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
from ...dialect.llh import DimOp, TorchSymbolicIntOp
from xdsl.irdl import IRDLOperation


@TORCH_FUNCTION_TRANSLATE("aten::clone","aten::alias","aten::detach")
def aten_clone_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    
    block: Block,
):
    value_map[node.name] = value_map[node.args[0].name]
    return None


@TORCH_FUNCTION_TRANSLATE("prims::clone")
def aten_clone_convert(
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    
    block: Block,
):
    value_map[node.name] = value_map[node.args[0].name]
    return None
