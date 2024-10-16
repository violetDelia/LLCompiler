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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_SYMBOLANALYSIS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_SYMBOLANALYSIS_H_

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>

#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::llh {

class SymbolAnalysis {
 public:
  explicit SymbolAnalysis(Operation *op);
  virtual ~SymbolAnalysis();
  static SymbolAnalysis *getInstance(Operation *op);
  static SymbolAnalysis *getInstance(Value value);

  SymbolicIntOp buildNewSymbolFrom(Value &value);
  SymbolicIntOp getOrBuildSymbolFrom(Value &value, std::string val);
  SymbolicIntOp getOrBuildConstSymbolFrom(Value &value, size_t val);
  Value &addEncoding(Value &value, size_t result_pos = 0);
  Value &addEncoding(Value &value, llvm::ArrayRef<llvm::StringRef> symbols,
                     size_t result_pos = 0);

  SymbolRelationsOp buildRelations(LLHPatternRewriter *builder,
                                   llvm::StringRef symbol,
                                   llvm::ArrayRef<llvm::StringRef> relations,
                                   AffineExpr expr);
  void debugPrintSymbols();

 private:
  Operation *_getMainFunc(Operation *op);
  void _insertOp(LLHPatternRewriter *builder, Operation *op,
                 Value &value) const;

 public:
  static llvm::StringRef UNKOW_SYMBOL;
  static bool symbol_enable;

 private:
  static std::mutex mutex_;

  std::map<Operation *, size_t> module_map_;
  std::map<std::string, Operation *> symbols_table_;
  int next_symbol_id_ = 0;
  int next_module_id_ = 0;
};

class SymbolAnalysisManager {
  friend SymbolAnalysis;
  static SymbolAnalysisManager *instance_;
  std::map<ModuleOp, SymbolAnalysis *> analysis_map_;
  static std::mutex mutex_;
};
}  // namespace mlir::llh
#endif  //  INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_SYMBOLANALYSIS_H_
