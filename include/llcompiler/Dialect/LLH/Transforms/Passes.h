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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_PASSES_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_PASSES_H_

#include <memory>

#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::llh {

std::unique_ptr<::mlir::Pass> createTransformLayoutToNHWCPass();
std::unique_ptr<::mlir::Pass> createLoadWeightPass();
std::unique_ptr<::mlir::Pass> createStoreConstToWeightPass();
std::unique_ptr<::mlir::Pass> createGenerateSymbolPass();
std::unique_ptr<::mlir::Pass> createOperationlegalizationPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"

}  // namespace mlir::llh

#endif  //  INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_PASSES_H_
