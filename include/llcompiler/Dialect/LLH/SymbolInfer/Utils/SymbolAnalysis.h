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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_SYMBOLINFER_UTILS_SYMBOLANALYSIS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_SYMBOLINFER_UTILS_SYMBOLANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "symengine/add.h"
#include "symengine/basic.h"
#include "symengine/cwrapper.h"
#include "symengine/dict.h"
#include "symengine/integer.h"
#include "symengine/symbol.h"
#include "symengine/symengine_rcp.h"
namespace mlir::llh {
using Symbol = SymEngine::RCP<const SymEngine::Basic>;

class SymbolAnalysis {
 public:
  static SymbolAnalysis *getInstance(Operation *op);
  static SymbolAnalysis *getInstance(Value value);
  static bool isExtraSymbolicInferOp(Operation *op);
  static bool isExtraSymbolAttrInferOp(Operation *op);
  static bool isExtraSymbolEncodingInferOp(Operation *op);
  static bool isSymbolicInferOp(Operation *op);
  static bool hasSymbolAttr(Operation *op);
  static bool hasSymbolAttr(Value value);
  static bool isConst(const llvm::StringRef name);
  static int64_t getIntValue(const llvm::StringRef name);
  static bool hasEncodingOrBind(Value value);
  static llvm::SmallVector<llvm::StringRef> getEncodingShapes(Value value);
  static Operation *getEncodingBindOp(Value value);
  static bool shapeIsSame(Value lhs, Value rhs);

 public:
  bool cleanCache();
  SymbolicIntOp buildNewSymbol(bool greater_zore = false);
  SymbolicIntOp buildNewSymbol(const Symbol symbol, AffineMap affine_map,
                               llvm::ArrayRef<llvm::StringRef> relations,
                               bool greater_zore = false);
  SymbolicIntOp getOrBuildSymbol(const llvm::StringRef val,
                                 bool greater_zore = false);
  SymbolicIntOp getOrBuildConstSymbol(int64_t val);
  SymbolicIntOp buildNewSymbolWithRelation(const llvm::StringRef relation_lhs,
                                           const llvm::StringRef relation_rhs,
                                           SymbolRelation relation_kind);
  SymbolRelationOp buildSymbolRelation(const llvm::StringRef symbol,
                                       const llvm::StringRef relation,
                                       SymbolRelation relation_kind);

  llvm::StringRef getOrBuildSymbolAttrFrom(Operation *op);
  llvm::StringRef getOrBuildSymbolAttrFrom(Value value);
  llvm::StringRef BuildSymbolAttrFrom(Operation *op);
  llvm::StringRef BuildSymbolAttrFrom(Value value);
  SymbolBindOp buildSymbolBindFromAttr(Value value);
  EncodingBindOp buildEncodingBindFrom(Value value);
  EncodingBindOp buildEncodingBindFrom(Value value,
                                       llvm::ArrayRef<llvm::StringRef> symbols);
  void buildEncodingBindFrom(Operation *op);
  void unloadEncoding(Value value);
  void unloadEncoding(Operation *op);
  Value addEncoding(Value value);
  Value addEncoding(Value value, llvm::ArrayRef<llvm::StringRef> symbols);

  bool replaceSymbol(const llvm::StringRef old_symbol,
                     const llvm::StringRef new_symbol);
  ModuleOp getSymbolModule() const;
  ModuleOp getRootModule() const;
  bool hasSymbol(const llvm::StringRef symbol) const;
  void debugPrintSymbols();
  Symbol getBasicSymbol(const llvm::StringRef symbol);

 private:
  explicit SymbolAnalysis(Operation *op);
  virtual ~SymbolAnalysis();
  Operation *_getMainFunc(Operation *op);

  bool _isConst(Operation *op);
  bool _isConst(Value value);
  bool _remove(llvm::StringRef symbol);
  llvm::StringRef _getSymbolAttr(Operation *op);
  llvm::StringRef _getSymbolAttr(Value value);
  SymbolicIntOp _insertNewSymbol(const llvm::StringRef symbol_name,
                                 LLHPatternRewriter *builder,
                                 bool greater_zore);
  SymbolicIntOp _insertNewSymbol(const llvm::StringRef symbol_name,
                                 LLHPatternRewriter *builder, bool greater_zore,
                                 const Symbol symbol);
  void _insertSymbolicIntOp(LLHPatternRewriter *builder, Operation *op) const;
  void _insertToSymbolModule(LLHPatternRewriter *builder, Operation *op) const;
  SymbolRelationMapOp _buildSymbolRelation(
      const llvm::StringRef symbol, AffineMap affine_map,
      llvm::ArrayRef<llvm::StringRef> relations);

 public:
  // 未知symbol的标记
  static llvm::StringRef UNKOW_SYMBOL;
  static bool symbol_enable;
  // symbol module 的名字
  static mlir::StringRef symbol_module_name;

 private:
  static std::mutex mutex_;
  std::map<Operation *, size_t> module_map_;
  std::map<std::string, SymbolicIntOp> symbol_op_table_;
  Operation *symbol_module_;
  std::atomic<int> next_symbol_id_ = 0;
  std::atomic<int> next_module_id_ = 0;

  // 记录已经创建的符号
  std::map<std::string, Symbol> symbol_table_;
  // 记录表达式的map[exp,symbol]，如果已经存在表达式，说明符号相等，就不会创建新符号。
  std::map<std::string, std::string> expressions_map_;
};

// 防止多个Module同时infersymbol
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
#endif  //  INCLUDE_LLCOMPILER_DIALECT_LLH_SYMBOLINFER_UTILS_SYMBOLANALYSIS_H_
