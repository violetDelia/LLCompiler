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

def LLH_AddOp : LLH_Op<"add"> {
    let summary = "add op";
    let description = [{
        add op;
    }];

    let arguments = (ins 
        LLH_Tensor:$lhs,
        LLH_Tensor:$rhs);
    let results = (outs LLH_Tensor);
}


#endif // LLH_OPS
