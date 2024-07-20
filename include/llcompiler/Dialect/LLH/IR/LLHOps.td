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

def LLH_ConstOp : LLH_Op<"weight">{
    let summary = "weight op";
    let description = [{
        weight op that will not a const in traning;
    }];

    let arguments = (ins 
        ElementsAttr:$value);
    let results = (outs LLH_Tensor);
}

def LLH_AndOp : LLH_Op<"and"> {//shape inference
    let arguments = (ins 
        LLH_BoolTensor:$lhs,
        LLH_BoolTensor:$rhs);
    let results = (outs LLH_BoolTensor);
}

def LLH_AbsOp : LLH_Op<"abs">{//shape inference
    let arguments = (ins 
        LLH_Tensor:$input);
    let results = (outs LLH_Tensor);
}

def LLH_AddOp : LLH_Op<"add"> {//shape inference
    let arguments = (ins 
        LLH_Tensor:$lhs,
        LLH_Tensor:$rhs);
    let results = (outs LLH_Tensor);
}

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

def LLH_CosOp : LLH_Op<"cos">{//shape inference
    let arguments = (
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_CoshOp : LLH_Op<"cosh">{//shape inference
    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_SinOp : LLH_Op<"sin">{//shape inference
    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_SinHOp : LLH_Op<"sinh">{//shape inference
    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_TanOp : LLH_Op<"tan">{//shape inference
    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_tanHOp : LLH_Op<"tanh">{//shape inference
    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
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
