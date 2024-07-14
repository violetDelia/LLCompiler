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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLH_ConstantOp : LLH_Op<"constant"> {
    let summary = "constant op";
    let description = [{
        the operation cantain the constant.
    }];

    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        $input attr-dict `:` type($value)
    }];

    let hasVerifier = 1;
}

#endif // LLH_OPS
