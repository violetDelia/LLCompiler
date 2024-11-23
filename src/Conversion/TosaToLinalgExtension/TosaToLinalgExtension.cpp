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
#include "Conversion/TosaToLinalgExtension/TosaToLinalgExtension.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "Dialect/LLH/IR/LLHOps.h"
#include "Dialect/Utility/RewritePattern.h"
#include "Support/Logger.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir {
#define GEN_PASS_DEF_CONVERTTOSATOLINALGEXTENSIONPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// legal func
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateConvertTosaToLinalgExtensionPassPatterns(
    TypeConverter& converter, RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configConvertTosaToLinalgExtensionPassTarget(ConversionTarget& target) {}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void initConvertTosaToLinalgExtensionPassTypeConverter(
    TypeConverter& converter) {
  auto shaped_repalce = [](ShapedType type) { return type; };
  auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
  converter.addConversion(ranked_tensor_replace);
  converter.addConversion(shaped_repalce);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
struct ConvertTosaToLinalgExtensionPass
    : impl::ConvertTosaToLinalgExtensionPassBase<
          ConvertTosaToLinalgExtensionPass> {
  using impl::ConvertTosaToLinalgExtensionPassBase<
      ConvertTosaToLinalgExtensionPass>::ConvertTosaToLinalgExtensionPassBase;
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void ConvertTosaToLinalgExtensionPass::runOnOperation() {
  LLC_RUN_IN_PASS
  ConversionTarget target(getContext());
  configConvertTosaToLinalgExtensionPassTarget(target);
  TypeConverter converter;
  initConvertTosaToLinalgExtensionPassTypeConverter(converter);
  RewritePatternSet patterns(&getContext());
  populateConvertTosaToLinalgExtensionPassPatterns(converter, patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
