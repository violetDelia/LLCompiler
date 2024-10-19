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
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
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
  static SymbolAnalysis *getInstance(Operation *op);
  static SymbolAnalysis *getInstance(Value value);
  bool cleanCache();
  static bool isExtraSymbolicInferOp(Operation *op);
  static bool isSymbolicInferOp(Operation *op);
  static bool hasSymbolAttr(Operation *op);
  static bool hasSymbolAttr(Value value);
  static llvm::StringRef getSymbolAttr(Operation *op);
  static llvm::StringRef getSymbolAttr(Value value);

  SymbolicIntOp buildNewSymbol();
  SymbolicIntOp getOrBuildSymbol(const llvm::StringRef val);
  SymbolicIntOp getOrBuildConstSymbol(size_t val);
  Value addEncoding(Value value, size_t result_pos = 0);
  Value addEncoding(Value value, llvm::ArrayRef<llvm::StringRef> symbols,
                    size_t result_pos = 0);

  llvm::StringRef getOrBuildSymbolAttrFrom(Operation *op);
  llvm::StringRef getOrBuildSymbolAttrFrom(Value value);
  SymbolRelationOp buildSymbolRelation(const llvm::StringRef symbol,
                                       const llvm::StringRef relation,
                                       SymbolRelation relation_kind);
  SymbolBinaryRelationOp buildSymbolRelation(const llvm::StringRef symbol,
                                             const llvm::StringRef relation_lhs,
                                             const llvm::StringRef relation_rhs,
                                             SymbolRelation relation_kind);

  bool replaceSymbol(const llvm::StringRef old_symbol,
                     const llvm::StringRef new_symbol);
  ModuleOp getSymbolModule() const;
  ModuleOp getRootModule() const;
  bool hasSymbol(const llvm::StringRef symbol) const;

  void debugPrintSymbols();

 private:
  explicit SymbolAnalysis(Operation *op);
  virtual ~SymbolAnalysis();
  Operation *_getMainFunc(Operation *op);
  void _insertInModule(LLHPatternRewriter *builder, Operation *op) const;
  void _insertToSymbolModule(LLHPatternRewriter *builder, Operation *op) const;
  bool _isConst(Operation *op);
  bool _isConst(Value value);
  bool _isConst(llvm::StringRef name);
  bool _remove(llvm::StringRef symbol);
  ModuleOp _getRootModule(Operation *op);
  ModuleOp _getRootModule(Value value);

 public:
  //未知symbol的标记
  static llvm::StringRef UNKOW_SYMBOL;
  //弃用
  static bool symbol_enable;
  //symbol module 的名字
  static mlir::StringRef symbol_module_name;

 private:
  static std::mutex mutex_;
  std::map<Operation *, size_t> module_map_;
  std::map<std::string, Operation *> symbols_table_;
  Operation *symbol_module_;
  std::atomic<int> next_symbol_id_ = 0;
  std::atomic<int> next_module_id_ = 0;

 private:
  // TODO:
  std::map<std::string, std::unordered_set<std::string>> EQ_table;
  std::map<std::string, std::unordered_set<std::string>> GT_table;
  std::map<std::string, std::unordered_set<std::string>> LT_table;
  std::map<std::string, std::unordered_set<std::string>> GE_table;
  std::map<std::string, std::unordered_set<std::string>> LE_table;
  std::map<std::string, std::unordered_set<std::string>> Add_table;
  std::map<std::string, std::unordered_set<std::string>> Sub_table;
  std::map<std::string, std::unordered_set<std::string>> FloorDiv_table;
  std::map<std::string, std::unordered_set<std::string>> Mul_table;
};

//防止多个Module同时infersymbol
class SymbolAnalysisManager {
  friend SymbolAnalysis;
  static SymbolAnalysisManager &getInstance();
  static std::mutex mutex_;

 private:
  SymbolAnalysisManager() = default;
  SymbolAnalysisManager(const SymbolAnalysisManager &other) = delete;
  SymbolAnalysisManager &operator=(const SymbolAnalysisManager &) = delete;

 private:
  std::map<ModuleOp, SymbolAnalysis *> analysis_map_;
};
}  // namespace mlir::llh
#endif  //  INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_SYMBOLANALYSIS_H_
