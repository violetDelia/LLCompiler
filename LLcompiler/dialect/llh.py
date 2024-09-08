from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import *
from xdsl.dialects.utils import *
from xdsl.ir import *
from typing import *
from xdsl.irdl import *
from xdsl.irdl.constraints import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import *
from xdsl.utils.exceptions import VerifyException


#python 验证速度比较慢
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


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "llh.constant"
    value = attr_def(DenseIntOrFPElementsAttr)
    result = result_def(TensorType)


@irdl_op_definition
class WeightOp(IRDLOperation):
    name = "llh.weight"
    weight_file = attr_def(StringAttr)
    result = result_def(TensorType)

@irdl_op_definition
class ConvBiasOp(IRDLOperation):
    name = "llh.conv_bias"
    X = operand_def(TensorType)
    W = operand_def(TensorType)
    B = operand_def(TensorType)
    dilations = attr_def(ArrayAttr)
    group = attr_def(IntegerAttr)
    kernel_shape = attr_def(ArrayAttr)
    pads = attr_def(ArrayAttr)
    stride = attr_def(ArrayAttr)
    result = result_def(TensorType)

LLH = Dialect("llh", [ConstantOp, WeightOp], [])
