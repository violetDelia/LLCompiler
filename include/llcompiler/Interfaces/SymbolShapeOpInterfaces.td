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
#ifndef INCLUDE_LLCOMPILER_INTERFACES_SYMBOLSHAPEOPINTERFACES_TD_
#define INCLUDE_LLCOMPILER_INTERFACES_SYMBOLSHAPEOPINTERFACES_TD_
include "mlir/IR/OpBase.td"

def SymbolicInferShapeOpInterface: OpInterface<"SymbolicInferShapeOpInterface"> {
  let description = [{
    符号shepa推导的接口.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    /*retTy=*/"::llvm::LogicalResult",/*methodName=*/ "inferSymbolicShape">
  ];
}

#endif // INCLUDE_LLCOMPILER_INTERFACES_SYMBOLSHAPEOPINTERFACES_TD_