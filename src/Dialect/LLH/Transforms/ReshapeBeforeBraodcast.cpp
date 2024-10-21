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

#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/Braocast.h"
#include "llcompiler/Dialect/LLH/Utils/CommonRewrite.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Interfaces/BraodcastableOpInterfaces.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_RESHAPEBEFOREBRAODCASTPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateReshapeBeforeBraodcastPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<SimplyBinaryOpReshape<AddOp>>(context);
  patterns.add<SimplyBinaryOpReshape<SubOp>>(context);
  patterns.add<SimplyBinaryOpReshape<DivOp>>(context);
  patterns.add<SimplyBinaryOpReshape<MulOp>>(context);
  patterns.add<SimplyBinaryOpReshape<MaxOp>>(context);
  patterns.add<SimplyBinaryOpReshape<MinOp>>(context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configReshapeBeforeBraodcastPassConversionTarget(
    ConversionTarget& target) {
  // target.addIllegalOp<llh::SymbolicBindOp>();
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct ReshapeBeforeBraodcastPass
    : llh::impl::ReshapeBeforeBraodcastPassBase<ReshapeBeforeBraodcastPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void ReshapeBeforeBraodcastPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateReshapeBeforeBraodcastPassPatterns(patterns);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  populateSymbolCanonicalizePatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
