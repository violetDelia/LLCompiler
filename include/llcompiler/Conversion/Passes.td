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

#ifndef LLC_CONVERSION_PASSES
#define LLC_CONVERSION_PASSES

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
#endif // LLC_CONVERSION_PASSES
