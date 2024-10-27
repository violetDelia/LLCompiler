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
//
#ifndef INCLUDE_LLCOMPILER_DIALECT_LLVMXTENSION_TRANSFORMS_PASSES_TD_
#define INCLUDE_LLCOMPILER_DIALECT_LLVMXTENSION_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AdaptEntryParmsForEnginePass: Pass<"adapt-entry-parms-for-engine","ModuleOp"> {
  let summary = "调整入口函数的参数, 方便执行引擎运行";
  let description = [{
  }];
  let dependentDialects = [ "::mlir::LLVM::LLVMDialect",
                            "::mlir::mhlo::MhloDialect"];
}


#endif // LLH_PASS