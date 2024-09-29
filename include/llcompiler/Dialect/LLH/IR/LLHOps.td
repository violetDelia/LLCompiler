//    Copyright 2024 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef LLH_OPS
#define LLH_OPS
include "llcompiler/Dialect/LLH/IR/LLHAttrs.td"
include "llcompiler/Dialect/LLH/IR/LLHConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "llcompiler/Interfaces/SymbolShapeOpInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
class LLH_Op<string mnemonic, list<Trait> traits = [Pure]> :
        Op<LLH_Dialect, mnemonic, traits>{
    let summary = cppNamespace#opName#" op";
    let description = [{
        $_name op;
    }];
}

class LLH_SymbolOp <string mnemonic, list<Trait> traits = []>:
        LLH_Op<mnemonic,traits#[DeclareOpInterfaceMethods<InferShapedTypeOpInterface>,DeclareOpInterfaceMethods<SymbolicInferShapeOpInterface>]>{}


class LLH_UnaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_SymbolOp<mnemonic,traits#[]>{
        let arguments = !con((ins
            OperandType:$input),
            attributes);
        let results = (outs 
            ResultType:$result);
}

class LLH_BinaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_SymbolOp<mnemonic,traits#[]>{
        let arguments = !con((ins
            OperandType:$lhs,
            OperandType:$rhs),
            attributes);
        let results = (outs 
            ResultType:$result);
}

class LLH_UnaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_UnaryOp<mnemonic,traits# [SameOperandsAndResultType, SameOperandsAndResultRank], OperandType, ResultType, attributes>;

class LLH_BinaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_BinaryOp<mnemonic,traits# [], OperandType, ResultType, attributes>;


def LLH_TorchSymbolicIntOp : LLH_Op<"torch_symbolic_int",[]>{
    let description = [{
        torch 框架的symbol信息, 已弃用
    }];
    let arguments = (ins 
       LLH_StringAttr:$sym_name);
    let results = (outs LLH_Int64);
}

def LLH_SymbolicBindOp: Op<LLH_Dialect,"symbolic_bind">{
    let description = [{
        torch 框架的symbol信息, 已弃用
    }];
    let arguments = (ins 
       LLH_Symbolic_Type:$operand ,
       Variadic<LLH_Int64>:$bind_symbols,
       AffineMapAttr:$expressions);
}

def LLH_SymbolicIntOp : LLH_Op<"symbolic_int",[Symbol]>{
    let description = [{
        symbol 信息
    }];
    let arguments = (ins 
       SymbolNameAttr:$sym_name);
}

def LLH_SymbolicCastOp: Op<LLH_Dialect,"symbolic_cast",[DeclareOpInterfaceMethods<CastOpInterface>]>{
    let arguments = (ins 
       LLH_Symbolic_Type:$operand);
    let results = (outs 
        LLH_Tensor);
}

def LLH_SymbolRelationsOp: LLH_Op<"symbol_relations",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let description = [{
        描述符号关系的op
    }];
    let arguments = (ins 
       SymbolRefAttr:$symbol,
       SymbolRefArrayAttr:$relations,
       AffineMapAttr:$expressions);
}

def LLH_AotOp : LLH_Op<"aot", []>{
    let arguments = (ins 
       SymbolNameAttr:$name,
       Variadic<LLH_Computable_Type>:$inputs);
    let results = (outs Variadic<LLH_Computable_Type>:$outputs);
}

def LLH_ConstantOp : LLH_SymbolOp<"constant", [ConstantLike]>{
    let arguments = (ins 
      LLH_ConstantAttr:$value);
    let results = (outs LLH_Computable_Type);
    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value),
        [{
            $_state.getOrAddProperties<Properties>().value = value;
            $_state.addTypes(value.getType());
        }]>,
        OpBuilder<(ins "IntegerAttr":$value),
        [{
            $_state.getOrAddProperties<Properties>().value = value;
            $_state.addTypes(value.getType());
        }]>,
        OpBuilder<(ins "FloatAttr":$value),
        [{
            $_state.getOrAddProperties<Properties>().value = value;
            $_state.addTypes(value.getType());
        }]>
    ];
    let hasFolder = 1;
    //let hasCanonicalizer = 1;
}

def LLH_WeightOp : LLH_SymbolOp<"weight">{
    let description = [{
        WeightOp is the constant weight only contain weight file;
    }];
    let arguments = (ins 
        LLH_StringAttr:$weight_file);
    let results = (outs 
        LLH_Tensor);
}

def LLH_DimOp : LLH_Op<"dim">{
    let arguments = (ins 
        LLH_Tensor:$input,
        LLH_Int64:$dim);
    let results = (outs 
        LLH_Int64);
}

def LLH_ReshapeOp : LLH_SymbolOp<"reshape",[]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        Variadic<LLH_Int64>:$shapes);
    let results = (outs 
        LLH_Tensor);
}


def LLH_ShapeOfOp : LLH_Op<"shape_of">{
    let arguments = (ins 
        LLH_Tensor:$input);
    let results = (outs 
        Variadic<LLH_Int64>);
}

def LLH_CatOp : LLH_SymbolOp<"cat",[]>{
    let arguments = (ins 
        Variadic<LLH_Tensor>:$inputs,
        I64Attr:$dim);
    let results = (outs 
        LLH_Tensor);
}

def LLH_ExpandOp : LLH_SymbolOp<"expand",[]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        Variadic<LLH_Int64>:$shapes);
    let results = (outs 
        LLH_Tensor);
}

def LLH_TransposeOp : LLH_UnaryOp<"transpose", [SameOperandsAndResultRank], LLH_Tensor, LLH_Tensor, (ins DenseI64ArrayAttr:$perms)>;
def LLH_DropOp : LLH_UnaryOp<"drop", [SameOperandsAndResultRank, SameOperandsAndResultType], LLH_Tensor, LLH_Tensor, (ins F64Attr:$p)>;
def LLH_FlattenOp : LLH_UnaryOp<"flatten", [], LLH_Tensor, LLH_Tensor, (ins LLH_Int64:$dim)>;
def LLH_AdaptiveAvgPoolOp : LLH_UnaryOp<"adaptive_average_pool", [], LLH_Tensor, LLH_Tensor>;


def LLH_ReluOp          : LLH_UnaryElementwiseOp<"relu", [], LLH_Tensor>;

def LLH_AddOp           : LLH_BinaryElementwiseOp<"add", [ResultsBroadcastableShape], LLH_Computable_Type>;
def LLH_MulOp           : LLH_BinaryElementwiseOp<"mul", [ResultsBroadcastableShape], LLH_Computable_Type>;
def LLH_DivOp           : LLH_BinaryElementwiseOp<"div", [ResultsBroadcastableShape], LLH_Computable_Type>;


def LLH_ConvBiasOp : LLH_SymbolOp<"conv_bias">{
    let arguments = (ins 
        LLH_FloatTensor:$X,
        LLH_FloatTensor:$W,
        LLH_FloatTensor:$B,
        DenseI64ArrayAttr:$dilation,
        DenseI64ArrayAttr:$kernel_shape,
        DenseI64ArrayAttr:$pad,
        DenseI64ArrayAttr:$stride,
        DefaultValuedAttr<I64Attr, "1">:$group,
        OptionalAttr<LLH_LayoutAttr>:$layout);
    let results = (outs LLH_FloatTensor);
}

def LLH_ConvOp : LLH_SymbolOp<"conv">{
    let arguments = (ins 
        LLH_FloatTensor:$X,
        LLH_FloatTensor:$W,
        DenseI64ArrayAttr:$dilation,
        DenseI64ArrayAttr:$kernel_shape,
        DenseI64ArrayAttr:$pad,
        DenseI64ArrayAttr:$stride,
        DefaultValuedAttr<I64Attr, "1">:$group,
        OptionalAttr<LLH_LayoutAttr>:$layout);
    let results = (outs LLH_FloatTensor);
}

def LLH_MatMulOp          : LLH_SymbolOp<"matmul">{
    let arguments = (ins
        LLH_Tensor:$lhs,
        LLH_Tensor:$rhs);
    let results = (outs 
        LLH_Tensor:$result);
}

def LLH_LayerNormOp     : LLH_SymbolOp<"layer_norm">{
    let arguments = (ins
        LLH_Tensor:$input,
        LLH_Tensor:$scale,
        LLH_Tensor:$bias,
        F64Attr:$epsilon,
        I64Attr:$axis);
    let results = (outs 
        LLH_Tensor:$result);
}

def LLH_BatchNormOp     : LLH_SymbolOp<"batch_norm">{
    let arguments = (ins
        LLH_Tensor:$input,
        LLH_Tensor:$scale,
        LLH_Tensor:$bias,
        LLH_Tensor:$input_mean,
        LLH_Tensor:$input_var,
        F64Attr:$epsilon,
        F64Attr:$momentum);
    let results = (outs 
        LLH_Tensor:$result);
}

def LLH_MaxPoolOp       : LLH_SymbolOp<"max_pool">{
    let arguments = (ins
        LLH_Tensor:$input,
        I1Attr:$ceil_mode,
        DenseI64ArrayAttr:$dilation,
        DenseI64ArrayAttr:$kernel_shape,
        DenseI64ArrayAttr:$pad,
        DenseI64ArrayAttr:$stride);
    let results = (outs 
        LLH_Tensor:$result);
}

// def LLH_CastOp          : LLH_UnaryOp<"cast", [Elementwise], LLH_Tensor, LLH_Tensor, (ins DefaultValuedAttr<SI64Attr, "1">:$saturate, SI64Attr:$to)>;

// def LLH_CeluOp        : LLH_UnaryElementwiseOp<"celu", [], LLH_FloatTensor, LLH_FloatTensor, (ins DefaultValuedAttr<F64Attr, "1">:$alpha)>;

// def LLH_AbsOp           : LLH_UnaryElementwiseOp<"abs", [], LLH_Tensor>;

// def LLH_AcosOp           : LLH_UnaryElementwiseOp<"acos", [], LLH_FloatTensor>;

// def LLH_AcoshOp          : LLH_UnaryElementwiseOp<"acosh", [], LLH_FloatTensor>;

// def LLH_AsinOp           : LLH_UnaryElementwiseOp<"asin", [], LLH_FloatTensor>;

// def LLH_AsinhOp          : LLH_UnaryElementwiseOp<"asinh", [], LLH_FloatTensor>;

// def LLH_AtanOp          : LLH_UnaryElementwiseOp<"atan", [], LLH_FloatTensor>;

// def LLH_AtanhOp         : LLH_UnaryElementwiseOp<"atanh", [], LLH_FloatTensor>;

// def LLH_CeilOp          : LLH_UnaryElementwiseOp<"ceil", [], LLH_FloatTensor>;

// def LLH_BitwiseNotOp    : LLH_BinaryElementwiseOp<"bitwise_not", [], LLH_Tensor, LLH_Tensor>;

// def LLH_AndOp           : LLH_BinaryElementwiseOp<"and", [], LLH_BoolTensor>;

// def LLH_AddOp           : LLH_BinaryElementwiseOp<"add", [Commutative], LLH_Tensor>;

// def LLH_BitShiftOp      : LLH_BinaryElementwiseOp<"bit_shift", [], LLH_IntTensor, LLH_IntTensor, (ins LLH_ShiftDirectionAttr:$direction)>;

// def LLH_BitwiseAndOp    : LLH_BinaryElementwiseOp<"bitwise_and", [], LLH_Tensor, LLH_Tensor>;

// def LLH_BitwiseOrOp     : LLH_BinaryElementwiseOp<"bitwise_or", [], LLH_Tensor, LLH_Tensor>;

// def LLH_BitwiseXorOp    : LLH_BinaryElementwiseOp<"bitwise_xor", [], LLH_Tensor, LLH_Tensor>;

// def LLH_CastLikeOp      : LLH_Op<"cast_like">{
//     let arguments = (ins 
//        LLH_Tensor:$input,
//        LLH_Tensor:$target_type,
//        DefaultValuedAttr<SI64Attr, "1">:$saturate);
//     let results = (outs LLH_Tensor);
// }

// def LLH_ArgMaxOp        : LLH_Op<"argmax", [InferShapedTypeOpInterface, SameOperandsAndResultType]>{
//     let arguments = (ins 
//         LLH_Tensor:$input,
//         DefaultValuedAttr<SI64Attr, "0">:$axis,
//         DefaultValuedAttr<SI64Attr, "1">:$keepdims,
//         DefaultValuedAttr<SI64Attr, "0">:$select_last_index);
//     let results = (outs LLH_Tensor);
// }

// def LLH_ArgMinOp        : LLH_Op<"argmin", [InferShapedTypeOpInterface, SameOperandsAndResultType]>{
//     let arguments = (ins 
//         LLH_Tensor:$input,
//         DefaultValuedAttr<SI64Attr, "0">:$axis,
//         DefaultValuedAttr<SI64Attr, "1">:$keepdims,
//         DefaultValuedAttr<SI64Attr, "0">:$select_last_index);
//     let results = (outs LLH_Tensor);
// }

// def LLH_AveragePoolOp   : LLH_Op<"average_pool", [InferShapedTypeOpInterface, SameOperandsAndResultType, SameOperandsAndResultRank]>{
//     let arguments = (ins 
//         LLH_FloatTensor:$X,
//         ArrayAttr:$dilations,
//         ArrayAttr:$kernel_shape,
//         ArrayAttr:$pads,
//         ArrayAttr:$strides,
//         DefaultValuedAttr<SI64Attr, "0">:$count_include_pad,
//         DefaultValuedAttr<LLH_AutoPadAttr, "AutoPad::DEFAULT">:$auto_pad,
//         DefaultValuedAttr<LLH_CeilModeAttr, "CeilMode::FLOOR">:$ceil_mode
//         );
//     let results = (outs LLH_FloatTensor);
// }

// def LLH_BatchNormalizationOp : LLH_Op<"batch_normal", [InferShapedTypeOpInterface]>{
//     let arguments = (ins 
//         LLH_FloatTensor:$x,
//         LLH_FloatTensor:$scale,
//         LLH_FloatTensor:$bals,
//         LLH_FloatTensor:$mean,
//         LLH_FloatTensor:$var,
//         DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
//         DefaultValuedAttr<F64Attr, "0.9">:$momentum,
//         DefaultValuedAttr<F64Attr, "0">:$training_mode
//         );
//     let results = (outs LLH_FloatTensor);
// }

// def LLH_ClipOp : LLH_Op<"clip", [InferShapedTypeOpInterface, SameOperandsAndResultRank]>{
//     let arguments = (ins 
//         LLH_Tensor:$input,
//         LLH_ScalarTensor:$min,
//         LLH_ScalarTensor:$max);
//     let results = (outs LLH_FloatTensor);
// }

// def LLH_CompressOp : LLH_Op<"compress", [InferShapedTypeOpInterface, SameOperandsAndResultRank]>{
//     let arguments = (ins 
//         LLH_Tensor:$input,
//         LLH_Rank1BoolTensor:$condition,
//         OptionalAttr<SI64Attr>:$axis);
//     let results = (outs LLH_FloatTensor);
// }


//Concat
//ConcatFromSequence
//ConstantOfShape
//Conv
//ConvInteger
//ConvTranspose
//Cos
//Cosh
//CumSum
//DFT
//DeformConv
//DepthToSpace
//DequantizeLinear
//Det
//Div
//Dropout
//DynamicQuantizeLinear
//Einsum
//Elu
//Equal
//Erf
//Exp
//Expand
//EyeLike
//Flatten
//Floor
//GRU
//Gather
//GatherElements
//GatherND
//Gelu
//Gemm
//GlobalAveragePool
//GlobalLpPool
//GlobalMaxPool
//Greater
//GreaterOrEqual
//GridSample
//GroupNormalization
//HammingWindow
//HannWindow
//HardSigmoid
//HardSwish
//Hardmax
//Identity
//If
//ImageDecoder
//InstanceNormalization
//IsInf
//IsNaN
//LRN
//LSTM
//LayerNormalization
//LeakyRelu
//Less
//LessOrEqual
//Log
//LogSoftmax
//Loop
//LpNormalization
//LpPool
//MatMul
//MatMulInteger
// /Max
//MaxPool
//MaxRoiPool
//MaxUnpool
//Mean
//MeanVarianceNormalization
//MelWeightMatrix
//Min
//Mish
//Mod
//Mul
//Multinomial
//Neg
//NegativeLogLikelihoodLoss
//NonMaxSuppression
//NonZero
//Not
//OneHot
//Optional
//OptionalGetElement
//OptionalHasElement
//Or
//PRelu
//Pad
//Pow
//QLinearConv
//QLinearMatMul
//QuantizeLinear
//RNN
//RandomNormal
//RandomNormalLike
//RandomUniform
//RandomUniformLike
//Range
//Reciprocal
//ReduceL1
//ReduceL2
//ReduceLogSum
//ReduceLogSumExp
//ReduceMax
//ReduceMean
//ReduceMin
//ReduceProd
//ReduceSum
//ReduceSumSquare
//RegexFullMatch
//Relu
//Reshape
//Resize
//ReverseSequence
//RoiAlign
//Round
//STFT
//Scan
//Scatter
//ScatterElements
//ScatterND
//Selu
//SequenceAt
//SequenceConstruct
//SequenceEmpty
//SequenceErase
//SequenceInsert
//SequenceLength
//SequenceMap
//Shape
//Shrink
//Sigmoid
//Sign
//Sin
//Sinh
//Size
//Slice
//Softmax
//SoftmaxCrossEntropyLoss
//Softplus
//Softsign
//SpaceToDepth
//Split
//SplitToSequence
//Sqrt
//Squeeze
//StringConcat
//StringNormalizer
//StringSplit
//Sub
//Sum
//Tan
//Tanh
//TfIdfVectorizer
//ThresholdedRelu
//Tile
//TopK
//Transpose
//Trilu
//Unique
//Unsqueeze
//Upsample
//Where
//Xor





//BernoulliOp
//BlackmanWindow
//CenterCropPad
//Col2Im
//AffineGrid
#endif // LLH_OPS
