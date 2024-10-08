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

#ifndef INCLUDE_LLCOMPILER_CONVERSION_PASSES_TD_
#define INCLUDE_LLCOMPILER_CONVERSION_PASSES_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LLH --> Tosa
//===----------------------------------------------------------------------===//
def ConvertLLHToTosaPass : Pass<"convert-llh-to-tosa"> {
  let summary = "convert LLH IR to Tosa IR";
  let description = [{

  }];
  let dependentDialects = [ "::mlir::tosa::TosaDialect",
                            "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// LLH --> Arith
//===----------------------------------------------------------------------===//
def ConvertLLHToArithPass : Pass<"convert-llh-to-arith"> {
  let summary = "convert LLH IR to Arith IR";
  let description = [{

  }];
  let dependentDialects = [ "::mlir::arith::ArithDialect",
                            "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// LLH --> Tensor
//===----------------------------------------------------------------------===//
def ConvertLLHToTensorPass : Pass<"convert-llh-to-tensor"> {
  let summary = "convert LLH IR to Tensor IR";
  let description = [{

  }];
  let dependentDialects = [ "::mlir::tensor::TensorDialect",
                            "::mlir::func::FuncDialect",
                            "::mlir::index::IndexDialect"];
}

#endif // INCLUDE_LLCOMPILER_CONVERSION_PASSES_TD_
