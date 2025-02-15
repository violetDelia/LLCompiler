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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/CommonRewrite.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_INSERTBROADCASTPASS
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

}  // namespace
using LLHAddOpBroadcastInsert = SimplyBinaryOpInsertBroadcast<AddOp>;
using LLHSubOpBroadcastInsert = SimplyBinaryOpInsertBroadcast<SubOp>;
using LLHDivOpBroadcastInsert = SimplyBinaryOpInsertBroadcast<DivOp>;
using LLHMulOpBroadcastInsert = SimplyBinaryOpInsertBroadcast<MulOp>;
using LLHMaxOpBroadcastInsert = SimplyBinaryOpInsertBroadcast<MaxOp>;
using LLHMinOpBroadcastInsert = SimplyBinaryOpInsertBroadcast<MinOp>;
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
using namespace mlir::llh::impl;
LLC_DEFINR_PASS(
    InsertBroadCast,
    {
      LLC_ADD_PATTERN(LLHAddOpBroadcastInsert);
      LLC_ADD_PATTERN(LLHSubOpBroadcastInsert);
      LLC_ADD_PATTERN(LLHDivOpBroadcastInsert);
      LLC_ADD_PATTERN(LLHMulOpBroadcastInsert);
      LLC_ADD_PATTERN(LLHMaxOpBroadcastInsert);
      LLC_ADD_PATTERN(LLHMinOpBroadcastInsert);
    },
    { populateSymbolCanonicalizePatterns(patterns); }, {})
