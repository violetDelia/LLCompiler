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
#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_TD_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_TD_
include "llcompiler/Dialect/LLH/IR/LLHAttrs.td"
include "llcompiler/Dialect/LLH/IR/LLHConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "llcompiler/Interfaces/BraodcastableOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "llcompiler/Interfaces/SymbolShapeOpInterfaces.td"


class LLH_Op<string mnemonic, list<Trait> traits = [Pure]> :
        Op<LLH_Dialect, mnemonic, traits>{
    let summary = cppNamespace#opName#" op";
    let description = [{
        $_name op;
    }];
}

class LLH_SymbolOp <string mnemonic, list<Trait> traits = []>:
        LLH_Op<mnemonic,traits#[DeclareOpInterfaceMethods<SymbolicInferShapeOpInterface>]>{}


class LLH_UnaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_SymbolOp<mnemonic,traits#[SameOperandsAndResultElementType, SameOperandsAndResultRank]>{
        let arguments = !con((ins
            OperandType:$input),
            attributes);
        let results = (outs 
            ResultType:$result);
}

class LLH_BinaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_SymbolOp<mnemonic,traits#[SameOperandsAndResultElementType, DeclareOpInterfaceMethods<BraodcastableOpInterface>]>{
        let arguments = !con((ins
            OperandType:$lhs,
            OperandType:$rhs),
            attributes);
        let results = (outs 
            ResultType:$result);
}

class LLH_UnaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_UnaryOp<mnemonic,traits# [], OperandType, ResultType, attributes>;

class LLH_BinaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        LLH_BinaryOp<mnemonic,traits# [], OperandType, ResultType, attributes>;


def LLH_TorchSymbolicIntOp : LLH_Op<"torch_symbolic_int",[]>{
    let description = [{
        torch 框架从fx_graph直接获取的符号信息。
    }];
    let arguments = (ins 
       LLH_StringAttr:$sym_name
       );
    let results = (outs LLH_Int64);
}

def LLH_SymbolicBindOp: Op<LLH_Dialect,"symbolic_bind">{
    let description = [{
        根据fake tensor 的信息,每创建一个op,就相应创建一个symbolic_bind将torch_symbolic_int与Op的结果绑定。
    }];
    let arguments = (ins 
       LLH_Symbolic_Type:$operand ,
       Variadic<LLH_Int64>:$bind_symbols,
       AffineMapAttr:$expressions);
}

def LLH_EncodingBindOp : Op<LLH_Dialect,"encoding_bind">{
    let description = [{
        将tensor的encoding信息绑到这个Op上,防止用标准的Pass或者第三方库的Pass出现不识别encoding导致的错误。
    }];
    let arguments = (ins
        LLH_Eencoding_Bind_Type:$operand ,
        LLH_Encoding:$encoding
    );
    let hasCanonicalizer = 1;
}

def LLH_SymbolBindOp : Op<LLH_Dialect, "symbol_bind">{
    let description = [{
        单符号
    }];
    let arguments = (ins 
       AnySignlessIntegerOrIndex:$operand ,
       FlatSymbolRefAttr:$symbol);
    let hasCanonicalizer = 1;
}


def LLH_SymbolicIntOp : LLH_Op<"symbolic_int",[Symbol]>{
    let description = [{
        symbol 信息
    }];
    let arguments = (ins 
       SymbolNameAttr:$sym_name//,
    //    DefaultValuedAttr<I64Attr, "1">:$min,
    //    DefaultValuedAttr<I64Attr, "INT64_MAX">:$max
       );
}

def LLH_SymbolicCastOp: Op<LLH_Dialect,"symbolic_cast",[DeclareOpInterfaceMethods<CastOpInterface>]>{
    let arguments = (ins 
       LLH_Symbolic_Type:$operand);
    let results = (outs 
        LLH_Tensor);
}

def LLH_SymbolRelationMapOp : LLH_Op<"symbol_relation_map",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let description = [{
        描述符号关系的op
    }];
    let arguments = (ins 
       FlatSymbolRefAttr:$symbol,
       SymbolRefArrayAttr:$relations,
       AffineMapAttr:$relation,
       LLH_StringAttr:$express
       );
}

def LLH_SymbolBinaryRelationOp : LLH_Op<"symbol_binary_relation",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let description = [{
        描述符号关系的op
    }];
    let arguments = (ins 
       FlatSymbolRefAttr:$symbol,
       FlatSymbolRefAttr:$relations_lhs,
       FlatSymbolRefAttr:$relations_rhs,
       LLH_SymbolRelationsAttr:$relation_kind
       );
    let hasCanonicalizer = 1;
}

def LLH_SymbolRelationOp: LLH_Op<"symbol_relation",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let description = [{
        描述符号关系的op
    }];
    let arguments = (ins 
       FlatSymbolRefAttr:$symbol,
       FlatSymbolRefAttr:$relation,
       LLH_SymbolRelationsAttr:$relation_kind
       );
    let hasCanonicalizer = 1;
}

def LLH_AotOp : LLH_Op<"aot", []>{
    let arguments = (ins 
       SymbolNameAttr:$name,
       Variadic<LLH_Computable_Type>:$inputs);
    let results = (outs Variadic<LLH_Computable_Type>:$outputs);
}

def LLH_ConstantOp : LLH_SymbolOp<"constant", [ConstantLike]>{
    let arguments = (ins 
      LLH_ConstantAttr:$value,
      OptionalAttr<FlatSymbolRefAttr>:$symbol);
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
    let hasCanonicalizer = 1;
}

def LLH_BroadCastToOp          : LLH_SymbolOp<"broadcast_to",[]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        Variadic<LLH_Int64>:$out_shapes,
        DenseI64ArrayAttr:$cast_dims,
        OptionalAttr<DenseI64ArrayAttr>:$expand_dims,
        OptionalAttr<DenseI64ArrayAttr>:$noexpand_dims
        );
    let results = (outs 
        LLH_Tensor);
    let hasCanonicalizer = 1;
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
        LLH_Int64:$dim,
        OptionalAttr<FlatSymbolRefAttr>:$symbol
        );
    let results = (outs 
        LLH_Int64);
    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$dim),
        [{
            llvm::SmallVector<mlir::Value> operands;
            auto const_dim = $_builder.create<mlir::llh::ConstantOp>(input.getLoc(),$_builder.getI64IntegerAttr(dim));
            operands.push_back(input);
            operands.push_back(const_dim);
            $_state.addOperands(operands);
            $_state.addTypes($_builder.getI64Type());
        }]>,
    ];
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def LLH_ExtractOp : LLH_SymbolOp<"extract",[]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        LLH_Int64:$index);
    let results = (outs 
        LLH_Computable_Type);
    let hasCanonicalizer = 1;
}

def LLH_StrideSliceOp : LLH_SymbolOp<"stride_slice",[AttrSizedOperandSegments]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        Variadic<LLH_Int64>:$start_index,
        Variadic<LLH_Int64>:$end_index,
        Variadic<LLH_Int64>:$strides);
    let results = (outs 
        LLH_Tensor);
}

def LLH_SliceOp : LLH_SymbolOp<"slice",[AttrSizedOperandSegments]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        Variadic<LLH_Int64>:$start_index,
        Variadic<LLH_Int64>:$end_index);
    let results = (outs 
        LLH_Tensor);
    let hasCanonicalizer = 1;
}

def LLH_ReshapeOp : LLH_SymbolOp<"reshape",[]>{
    let arguments = (ins 
        LLH_Tensor:$input,
        Variadic<LLH_Int64>:$shapes);
    let results = (outs 
        LLH_Tensor);
    let hasCanonicalizer = 1;
}

def LLH_EmptyOp : LLH_SymbolOp<"empty",[NoMemoryEffect]>{
    let arguments = (ins 
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

def LLH_FlattenOp : LLH_SymbolOp<"flatten", []>{
    let arguments = (ins 
        LLH_Tensor:$input,
        LLH_Int64:$dim);
    let results = (outs 
        LLH_Tensor);
}

def LLH_TransposeOp : LLH_UnaryOp<"transpose", [SameOperandsAndResultRank], LLH_Tensor, LLH_Tensor, (ins DenseI64ArrayAttr:$perms)>;

def LLH_DropOp : LLH_UnaryOp<"drop", [SameOperandsAndResultRank, SameOperandsAndResultType], LLH_Tensor, LLH_Tensor, (ins F64Attr:$p)>;

def LLH_ConvertToOp : LLH_UnaryOp<"convert_to", [SameOperandsAndResultRank], LLH_Tensor, LLH_Tensor>;

def LLH_AdaptiveAvgPoolOp : LLH_UnaryOp<"adaptive_average_pool", [], LLH_Tensor, LLH_Tensor, (ins DenseI64ArrayAttr:$out_size)>{
    let hasCanonicalizer = 1;
}


def LLH_ReluOp          : LLH_UnaryElementwiseOp<"relu", [], LLH_Tensor>;

def LLH_AbsOp          : LLH_UnaryElementwiseOp<"abs", [], LLH_Tensor>{
    let hasCanonicalizer = 1;
}

def LLH_SqrtOp         : LLH_UnaryElementwiseOp<"sqrt", [], LLH_Tensor>;

def LLH_RsqrtOp         : LLH_UnaryElementwiseOp<"rsqrt", [], LLH_Tensor>;

def LLH_AddOp           : LLH_BinaryElementwiseOp<"add", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{
    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def LLH_MulOp           : LLH_BinaryElementwiseOp<"mul", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{
    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def LLH_DivOp           : LLH_BinaryElementwiseOp<"div", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{
    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def LLH_SubOp           : LLH_BinaryElementwiseOp<"sub", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{
    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def LLH_MaxOp           : LLH_BinaryElementwiseOp<"max", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def LLH_MinOp           : LLH_BinaryElementwiseOp<"min", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def LLH_CompareOp           : LLH_SymbolOp<"compare", [ResultsBroadcastableShape]>{
    let arguments = (ins 
        LLH_Tensor:$lhs,
        LLH_Tensor:$rhs,
        LLH_CompareAttr:$kind,
        OptionalAttr<FlatSymbolRefAttr>:$symbol
        );
    let results = (outs 
        LLH_Tensor);
    let hasCanonicalizer = 1;
}

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
        OptionalAttr<LLH_LayoutAttr>:$layout,
        OptionalAttr<LLH_LayoutAttr>:$weight_layout);
    let results = (outs LLH_FloatTensor);
    let hasCanonicalizer = 1;
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
        OptionalAttr<LLH_LayoutAttr>:$layout,
        OptionalAttr<LLH_LayoutAttr>:$weight_layout);
    let results = (outs LLH_FloatTensor);
    let hasCanonicalizer = 1;
}

def LLH_MatMulOp          : LLH_SymbolOp<"matmul",[DeclareOpInterfaceMethods<BraodcastableOpInterface>]>{
    let arguments = (ins
        LLH_2DTensor:$lhs,
        LLH_2DTensor:$rhs);
    let results = (outs 
        LLH_2DTensor:$result);
}

def LLH_BatchMatMulOp          : LLH_SymbolOp<"batch_matmul",[DeclareOpInterfaceMethods<BraodcastableOpInterface>]>{
    let arguments = (ins
        LLH_3DTensor:$lhs,
        LLH_3DTensor:$rhs);
    let results = (outs 
        LLH_3DTensor:$result);
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
        F64Attr:$momentum,
        I64Attr:$feature_index);
    let results = (outs 
        LLH_Tensor:$result,
        LLH_Tensor:$running_mean,
        LLH_Tensor:$running_var);
}

def LLH_BatchNormInferenceOp     : LLH_SymbolOp<"batch_norm_inference">{
    let arguments = (ins
        LLH_Tensor:$input,
        LLH_Tensor:$scale,
        LLH_Tensor:$bias,
        LLH_Tensor:$input_mean,
        LLH_Tensor:$input_var,
        F64Attr:$epsilon,
        F64Attr:$momentum,
        I64Attr:$feature_index);
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
        DenseI64ArrayAttr:$stride,
        OptionalAttr<LLH_LayoutAttr>:$layout);
    let results = (outs 
        LLH_Tensor:$result);
    let hasCanonicalizer = 1;
}

def LLH_WhereOp : LLH_SymbolOp<"where", []>{
    let arguments = (ins 
        LLH_BoolTensor:$pred,
        LLH_Tensor:$on_true,
        LLH_Tensor:$on_false);
    let results = (outs 
        LLH_Tensor);
}


#endif // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_TD_
