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

# python 验证速度比较慢
i8 = IntegerType(8)

LLH_Bool: TypeAlias = I1
LLH_Int8: TypeAlias = Annotated[IntegerType, i8]
LLH_Int16: TypeAlias = I16
LLH_Int32: TypeAlias = I32
LLH_Int64: TypeAlias = I64

LLH_UInt8: TypeAlias = Annotated[IntegerType, IntegerType(8, Signedness.UNSIGNED)]
LLH_UInt16: TypeAlias = Annotated[IntegerType, IntegerType(16, Signedness.UNSIGNED)]
LLH_UInt32: TypeAlias = Annotated[IntegerType, IntegerType(32, Signedness.UNSIGNED)]
LLH_UInt64: TypeAlias = Annotated[IntegerType, IntegerType(64, Signedness.UNSIGNED)]

LLH_SInt: TypeAlias = AnySignlessIntegerType
LLH_UInt: TypeAlias = Annotated[
    IntegerType,
    ParamAttrConstraint(IntegerType, [IntAttr, SignednessAttr(Signedness.SIGNLESS)]),
]
LLH_Int: TypeAlias = IntegerType

LLH_F16: TypeAlias = Float64Type
LLH_F32: TypeAlias = Float32Type
LLH_F64: TypeAlias = Float64Type
LLH_BF16: TypeAlias = BFloat16Type

LLH_Float = AnyFloat

LLH_BoolTensor: TypeAlias = TensorType[LLH_Bool]

LLH_Int8Tensor: TypeAlias = TensorType[LLH_Int8]
LLH_Int16Tensor: TypeAlias = TensorType[LLH_Int16]
LLH_Int32Tensor: TypeAlias = TensorType[LLH_Int32]
LLH_Int64Tensor: TypeAlias = TensorType[LLH_Int64]


LLH_UInt8Tensor: TypeAlias = TensorType[LLH_UInt8]
LLH_UInt16Tensor: TypeAlias = TensorType[LLH_UInt16]
LLH_UInt32Tensor: TypeAlias = TensorType[LLH_UInt32]
LLH_UInt64Tensor: TypeAlias = TensorType[LLH_UInt64]

LLH_SIntTensor: TypeAlias = TensorType[LLH_SInt]
LLH_UIntTensor: TypeAlias = TensorType[LLH_UInt]
LLH_IntTensor: TypeAlias = TensorType[LLH_Int]

LLH_F16Tensor: TypeAlias = TensorType[LLH_F16]
LLH_F32Tensor: TypeAlias = TensorType[LLH_F32]
LLH_F64Tensor: TypeAlias = TensorType[LLH_F64]
LLH_BF16Tensor: TypeAlias = TensorType[LLH_BF16]

LLH_FloatTensor: TypeAlias = TensorType[LLH_Float]

LLH_Tensor: TypeAlias = TensorType

LLH_Symbolic_Type = ContainerOf(AnyOf([TensorType, IntegerType]))
LLH_Computable_Type = ContainerOf(
    AnyOf(
        [TensorType, IntegerType, Float16Type, Float32Type, Float64Type, BFloat16Type]
    )
)


class CompareEnum(StrEnum):
    EQ = "EQ"
    LE = "LE"
    LT = "LT"
    GE = "GE"
    GT = "GT"
    NE = "NE"


class CompareAttribute(BitEnumAttribute[CompareEnum]):
    none_value = "none"
    all_value = "all"


@irdl_attr_definition
class CompareAttr(CompareAttribute):
    name = "llh.Compare"


class ModeEnum(StrEnum):
    Training = "training"
    Inference = "inference"


@irdl_attr_definition
class ModeAttr(BitEnumAttribute[ModeEnum]):
    name = "llh.Mode"
    none_value = "none"
    all_value = "all"
