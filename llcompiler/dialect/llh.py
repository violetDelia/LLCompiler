from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    I1,
    I16,
    I64,
    I32,
    Signedness,
    Float16Type,
    Float32Type,
    FloatAttr,
    Float64Type,
    AnyFloat,
    TensorType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    AnySignlessIntegerType,
    AnyTensorType,
    AnySignlessIntegerOrIndexType,
    AnyFloatConstr,
    SignednessAttr,
    IntAttr,
    BFloat16Type,
    ArrayAttr,
    StringAttr,
    SymbolRefAttr,
    AffineMapAttr,
    ContainerOf,
    BoolAttr,
    SymbolNameAttr,
)
from enum import auto
from xdsl.ir.affine.affine_expr import AffineExpr
from xdsl.ir import Dialect, BitEnumAttribute
from xdsl.irdl import (
    AnyOf,
    ConstraintVar,
    result_def,
    attr_def,
    opt_attr_def,
    OpResult,
    var_operand_def,
    irdl_attr_definition,
    operand_def,
    irdl_op_definition,
    var_result_def,
    IRDLOperation,
    ParameterDef,
    ParametrizedAttribute,
    TypeVar,
    AttrSizedOperandSegments,
)
from xdsl.utils.str_enum import StrEnum
from xdsl.irdl.constraints import ParamAttrConstraint, AnyOf
from typing import TypeAlias, Annotated
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.traits import SymbolTable
from dataclasses import dataclass
from .llh_constraint import *

@irdl_op_definition
class AOTOp(IRDLOperation):
    name = "llh.aot"
    name = attr_def(StringAttr)
    inputs = var_operand_def(LLH_Computable_Type)
    outputs = var_result_def(LLH_Computable_Type)


@irdl_op_definition
class TorchSymbolicIntOp(IRDLOperation):
    name = "llh.torch_symbolic_int"
    sym_name = attr_def(StringAttr)
    result = result_def(IntegerType)


@irdl_op_definition
class SymbolicBindOp(IRDLOperation):
    name = "llh.symbolic_bind"
    operand = operand_def(LLH_Symbolic_Type)
    bind_symbols = var_operand_def(IntegerType)
    expressions = attr_def(AffineMapAttr)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "llh.constant"
    value = attr_def(DenseIntOrFPElementsAttr)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class WeightOp(IRDLOperation):
    name = "llh.weight"
    weight_file = attr_def(StringAttr)
    result = result_def(TensorType)


@irdl_op_definition
class AbsOp(IRDLOperation):
    name = "llh.abs"
    input = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class ConvertToOp(IRDLOperation):
    name = "llh.convert_to"
    input = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class SqrtOp(IRDLOperation):
    name = "llh.sqrt"
    input = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class AddOp(IRDLOperation):
    name = "llh.add"
    lhs = operand_def(LLH_Computable_Type)
    rhs = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class SubOp(IRDLOperation):
    name = "llh.sub"
    lhs = operand_def(LLH_Computable_Type)
    rhs = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class DivOp(IRDLOperation):
    name = "llh.div"
    lhs = operand_def(LLH_Computable_Type)
    rhs = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class MulOp(IRDLOperation):
    name = "llh.mul"
    lhs = operand_def(LLH_Computable_Type)
    rhs = operand_def(LLH_Computable_Type)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class CompareOp(IRDLOperation):
    name = "llh.compare"
    lhs = operand_def(LLH_Computable_Type)
    rhs = operand_def(LLH_Computable_Type)
    kind = attr_def(CompareAttr)
    result = result_def(LLH_Computable_Type)


@irdl_op_definition
class ConvBiasOp(IRDLOperation):
    name = "llh.conv_bias"
    input = operand_def(TensorType)
    weight = operand_def(TensorType)
    bias = operand_def(LLH_Computable_Type)
    dilations = attr_def(ArrayAttr)
    kernel_shape = attr_def(ArrayAttr)
    pads = attr_def(ArrayAttr)
    stride = attr_def(ArrayAttr)
    group = attr_def(IntegerAttr)
    result = result_def(TensorType)


@irdl_op_definition
class ConvOp(IRDLOperation):
    name = "llh.conv"
    X = operand_def(TensorType)
    W = operand_def(TensorType)
    dilation = attr_def(ArrayAttr)
    kernel_shape = attr_def(ArrayAttr)
    pad = attr_def(ArrayAttr)
    stride = attr_def(ArrayAttr)
    group = attr_def(IntegerAttr)
    result = result_def(TensorType)


@irdl_op_definition
class MatmulOp(IRDLOperation):
    name = "llh.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)


@irdl_op_definition
class BatchMatmulOp(IRDLOperation):
    name = "llh.batch_matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)


@irdl_op_definition
class TransposeOp(IRDLOperation):
    name = "llh.transpose"
    input = operand_def(TensorType)
    perms = attr_def(ArrayAttr)
    result = result_def(TensorType)


@irdl_op_definition
class DimOp(IRDLOperation):
    name = "llh.dim"
    input = operand_def(TensorType)
    dim = operand_def(IntegerType)
    result = result_def(IntegerType)


@irdl_op_definition
class SliceOp(IRDLOperation):
    name = "llh.slice"
    input = operand_def(TensorType)
    starts = var_operand_def(IntegerType)
    ends = var_operand_def(IntegerType)
    strides = var_operand_def(IntegerType)
    result = result_def(TensorType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class ExtractOp(IRDLOperation):
    name = "llh.extract"
    input = operand_def(TensorType)
    index = operand_def(IntegerType)
    result = result_def(TensorType)


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    name = "llh.reshape"
    input = operand_def(TensorType)
    shapes = var_operand_def(IntegerType)
    result = result_def(TensorType)


@irdl_op_definition
class BroadCastToOp(IRDLOperation):
    name = "llh.broadcast_to"
    input = operand_def(TensorType)
    out_shapes = var_operand_def(IntegerType)
    cast_dims = attr_def(ArrayAttr)
    expand_dims = opt_attr_def(ArrayAttr)
    noexpand_dims = opt_attr_def(ArrayAttr)
    result = result_def(TensorType)


@irdl_op_definition
class EmptyOp(IRDLOperation):
    name = "llh.empty"
    shapes = var_operand_def(IntegerType)
    result = result_def(TensorType)


@irdl_op_definition
class ExpandOp(IRDLOperation):
    name = "llh.expand"
    input = operand_def(TensorType)
    shape = var_operand_def(IntegerType)
    result = result_def(TensorType)


@irdl_op_definition
class FlattenOp(IRDLOperation):
    name = "llh.flatten"
    input = operand_def(TensorType)
    dim = operand_def(IntegerType)
    result = result_def(TensorType)


@irdl_op_definition
class shapeOfOp(IRDLOperation):
    name = "llh.shape"
    input = operand_def(TensorType)
    results = var_result_def(IntegerType)


@irdl_op_definition
class CatOp(IRDLOperation):
    name = "llh.cat"
    inputs = var_operand_def(TensorType)
    dim = attr_def(IntegerAttr)
    result = result_def(TensorType)


@irdl_op_definition
class DropOp(IRDLOperation):
    name = "llh.drop"
    input = operand_def(TensorType)
    p = attr_def(FloatAttr)
    result = result_def(TensorType)


@irdl_op_definition
class LayerNormOp(IRDLOperation):
    name = "llh.layer_norm"
    input = operand_def(TensorType)
    scale = operand_def(TensorType)
    bias = operand_def(TensorType)
    epsilon = attr_def(FloatAttr)
    axis = attr_def(IntegerAttr)
    result = result_def(TensorType)


@irdl_op_definition
class BatchNormOp(IRDLOperation):
    name = "llh.batch_norm"
    input = operand_def(TensorType)
    scale = operand_def(TensorType)
    bias = operand_def(TensorType)
    input_mean = operand_def(TensorType)
    input_var = operand_def(TensorType)
    epsilon = attr_def(FloatAttr)
    momentum = attr_def(FloatAttr)
    feature_index = attr_def(IntAttr)
    result = result_def(TensorType)
    running_mean = result_def(TensorType)
    running_var = result_def(TensorType)

@irdl_op_definition
class BatchNormInferenceOp(IRDLOperation):
    name = "llh.batch_norm_inference"
    input = operand_def(TensorType)
    scale = operand_def(TensorType)
    bias = operand_def(TensorType)
    input_mean = operand_def(TensorType)
    input_var = operand_def(TensorType)
    epsilon = attr_def(FloatAttr)
    momentum = attr_def(FloatAttr)
    feature_index = attr_def(IntAttr)
    result = result_def(TensorType)

@irdl_op_definition
class ReluOp(IRDLOperation):
    name = "llh.relu"
    input = operand_def(TensorType)
    result = result_def(TensorType)


@irdl_op_definition
class MaxPoolOp(IRDLOperation):
    name = "llh.max_pool"
    input = operand_def(TensorType)
    ceil_mode = attr_def(BoolAttr)
    dilation = attr_def(ArrayAttr)
    kernel_shape = attr_def(ArrayAttr)
    pad = attr_def(ArrayAttr)
    stride = attr_def(ArrayAttr)
    result = result_def(TensorType)


@irdl_op_definition
class AdaptiveAvgPoolOp(IRDLOperation):
    name = "llh.adaptive_average_pool"
    input = operand_def(TensorType)
    out_size = attr_def(ArrayAttr)
    result = result_def(TensorType)


LLH = Dialect(
    "llh",
    [
        ConstantOp,
        WeightOp,
        ConvBiasOp,
        ConvOp,
        AddOp,
        DivOp,
        MatmulOp,
        TransposeOp,
        MulOp,
        DimOp,
        shapeOfOp,
        ExpandOp,
        CatOp,
        DropOp,
        LayerNormOp,
        ReluOp,
        MaxPoolOp,
        AdaptiveAvgPoolOp,
        FlattenOp,
        BatchNormOp,
        SubOp,
        EmptyOp,
        AbsOp,
        SliceOp,
        ExtractOp,
        BroadCastToOp,
        BatchMatmulOp,
        SqrtOp,
        CompareOp,
        ConvertToOp,
        BatchNormInferenceOp,
    ],
    [],
)
