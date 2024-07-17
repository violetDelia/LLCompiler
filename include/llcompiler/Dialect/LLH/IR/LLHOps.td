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

def LLH_WeightOp : LLH_Op<"weight">{
    let summary = "weight op";
    let description = [{
        weight op that will not a const in traning;
    }];

    let arguments = (ins 
        LLH_Tensor:$shape,
        LLH_Tensor:$value);
    let results = (outs LLH_Tensor);
}

def LLH_AbsOp : LLH_Op<"abs">{//shape inference
    let summary = "abs op";
    let description = [{
        abs op
    }];

    let arguments = (ins 
        LLH_Tensor:$input);
    let results = (outs LLH_Tensor);
}

def LLH_AcosOp : LLH_Op<"acos">{//shape inference
    let summary = "acos op";
    let description = [{
        acos op
    }];

    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_AcoshOp : LLH_Op<"acosh">{//shape inference
    let summary = "acosh op";
    let description = [{
        acosh op
    }];

    let arguments = (ins 
        LLH_FloatTensor:$input);
    let results = (outs LLH_FloatTensor);
}

def LLH_AddOp : LLH_Op<"add"> {//shape inference
    let summary = "add op";
    let description = [{
        add op;
    }];

    let arguments = (ins 
        LLH_Tensor:$lhs,
        LLH_Tensor:$rhs);
    let results = (outs LLH_Tensor);
}

def LLH_AndOp : LLH_Op<"and"> {//shape inference
    let summary = "and op";
    let description = [{
        and op;
    }];

    let arguments = (ins 
        LLH_BoolTensor:$lhs,
        LLH_BoolTensor:$rhs);
    let results = (outs LLH_BoolTensor);
}

#endif // LLH_OPS
