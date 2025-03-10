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
#include "llcompiler/Conversion/LLHToFunc/LLHToFunc.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLHTOFUNCPASS
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
func::FuncOp lookupOrCreateFn(Operation* module, StringRef name,
                              ArrayRef<Type> input_types,
                              ArrayRef<Type> result_types,
                              bool sym_private = true) {
  CHECK(llc::MLIR, module->hasTrait<OpTrait::SymbolTable>());
  auto func = llvm::dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, name));
  if (func) return func;
  auto builder = OpBuilder(module->getRegion(0));
  auto func_op = builder.create<func::FuncOp>(
      module->getLoc(), name,
      FunctionType::get(module->getContext(), input_types, result_types));
  if (sym_private) func_op.setPrivate();
  return func_op;
}
//===----------------------------------------------------------------------===//
// legal func
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct LLHHostPrintOpToFunc : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;

  LogicalResult match(PrintOp op) const {
    if (!(op->getParentOfType<gpu::GPUModuleOp>() == nullptr)) {
      return llvm::failure();
    }
    return llvm::success();
  }

  void rewrite(PrintOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    auto input = op.getInput();
    auto input_type = input.getType();
    auto description = op.getPrefixDescription();
    rewriter.create<vector::PrintOp>(loc, description );
    rewriter.create<vector::PrintOp>(loc, "\n" );
    if (auto memref_type = llvm::cast_or_null<MemRefType>(input_type)) {
      auto element_type = memref_type.getElementType();
      auto memref_space = memref_type.getMemorySpaceAsInt();
      auto unranked_memref =
          UnrankedMemRefType::get(element_type, memref_space);
      auto cast = rewriter.create<memref::CastOp>(loc, unranked_memref, input);
      func::FuncOp print;
      if (element_type.isF32()) {
        print = lookupOrCreateFn(op->getParentOfType<ModuleOp>(),
                                 "printMemrefF32", {unranked_memref}, {});
      } else {
        UNIMPLEMENTED(llc::MLIR_PASS) << "Unimplemented type";
      }
      rewriter.create<func::CallOp>(loc, print, cast->getOpResult(0));
      rewriter.eraseOp(op);
      return;
    }
  }

};  // namespace}
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

LLC_DEFINE_CONVERSION_PASS(
    ConvertLLHToFunc, { LLC_ADD_CONVERSION(LLHHostPrintOpToFunc); },
    {
      target.addLegalDialect<mlir::arith::ArithDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::index::IndexDialect>();
      target.addLegalDialect<mlir::LLVM::LLVMDialect>();
      target.addLegalDialect<mlir::memref::MemRefDialect>();
      target.addLegalDialect<mlir::vector::VectorDialect>();
    },
    {
      auto shaped_repalce = [](ShapedType type) { return type; };
      converter.addConversion(shaped_repalce);
    })
