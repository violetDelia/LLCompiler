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
include "llcompiler/Dialect/LLH/IR/LLHTypes.td"
include "llcompiler/Dialect/LLH/IR/LLHTypeConstraints.td"
include "llcompiler/Dialect/LLH/IR/LLHEnums.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLH_UndefinedOp : LLH_Op<"undefined"> {
    let summary = "undefined op";
    let description = [{
        undefined op;
    }];

    let arguments = (ins LLH_String:$name);
    let results = (outs);
}

class LLH_ConstOp <string mnemonic, list<Trait> traits>:
        LLH_Op<mnemonic,traits>{
    let arguments = (ins 
       DenseElementsAttr:$value);
    let results = (outs LLH_Tensor);
    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value),
        [{
            $_state.getOrAddProperties<Properties>().value = value;
            $_state.addTypes(value.getType());
        }]>
    ];
    //let hasFolder = 1;
    //let hasCanonicalizer = 1;
}

def LLH_ConstantOp : LLH_ConstOp<"constant">{
    let description = [{
        ConstantOp is a const that will never changed;
        Example:
        ```mlir
        %output = llh.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
        ```
    }];
}

def LLH_WeightOp : LLH_ConstOp<"weight">{
    let description = [{
        WeightOp is the weight of model and it will change when traning;
        Example:
        ```mlir
        %output = llh.weight dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
        ```
    }];
}

class LLH_UnaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType>:
        LLH_Op<mnemonic,traits#[InferShapedTypeOpInterface]>{
        let arguments = (ins 
            OperandType:$input);
        let results = (outs 
            ResultType:$result);
}

class LLH_BinaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType>:
        LLH_Op<mnemonic,traits#[InferShapedTypeOpInterface]>{
        let arguments = (ins 
            OperandType:$lhs,
            OperandType:$rhs);
        let results = (outs 
            ResultType:$result);
}

class LLH_UnaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType>:
        LLH_UnaryOp<mnemonic,traits# [SameOperandsAndResultType, SameOperandsAndResultRank], OperandType, ResultType>;

class LLH_BinaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType>:
        LLH_BinaryOp<mnemonic,traits# [SameOperandsAndResultType, SameOperandsAndResultRank], OperandType, ResultType>;

def LLH_CosOp   : LLH_UnaryElementwiseOp<"cos", [], LLH_FloatTensor>;

def LLH_CoshOp  : LLH_UnaryElementwiseOp<"cosh", [], LLH_FloatTensor>;

def LLH_SinOp   : LLH_UnaryElementwiseOp<"sin", [], LLH_FloatTensor>;

def LLH_SinHOp  : LLH_UnaryElementwiseOp<"sinh", [], LLH_FloatTensor>;

def LLH_TanOp   : LLH_UnaryElementwiseOp<"tan", [], LLH_FloatTensor>;

def LLH_tanHOp  : LLH_UnaryElementwiseOp<"tanh", [], LLH_FloatTensor>;

def LLH_AbsOp   : LLH_UnaryElementwiseOp<"abs", [], LLH_Tensor>;

def LLH_AndOp   : LLH_BinaryElementwiseOp<"and", [], LLH_BoolTensor>;

def LLH_AddOp   : LLH_BinaryElementwiseOp<"add", [], LLH_Tensor>;

def LLH_ArgMax : LLH_Op<"argmax">{//shape inference
    let arguments = (ins 
        LLH_Tensor:$input,
        DefaultValuedAttr<SI64Attr, "0">:$axis,
        DefaultValuedAttr<SI64Attr, "1">:$keepdims,
        DefaultValuedAttr<SI64Attr, "0">:$select_last_index);
    let results = (outs LLH_Tensor);
}

def LLH_ArgMin : LLH_Op<"argmin">{//shape inference
    let arguments = (ins 
        LLH_Tensor:$input,
        DefaultValuedAttr<SI64Attr, "0">:$axis,
        DefaultValuedAttr<SI64Attr, "1">:$keepdims,
        DefaultValuedAttr<SI64Attr, "0">:$select_last_index);
    let results = (outs LLH_Tensor);
}

def LLH_AveragePoolOp : LLH_Op<"average_pool">{//shape inference
    let arguments = (ins 
        LLH_Tensor:$input,
        //I64ElementsAttr:
        DefaultValuedAttr<LLH_AutoPadAttr, "AutoPad::DEFAULT">:$auto_pad,
        DefaultValuedAttr<LLH_CeilModeAttr, "CeilMode::FLOOR">:$ceil_mode
        );
    let results = (outs LLH_Tensor);
}

#endif // LLH_OPS
