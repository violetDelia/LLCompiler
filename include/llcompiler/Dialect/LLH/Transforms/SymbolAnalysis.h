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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_SYMBOLANALYSIS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_SYMBOLANALYSIS_H_

#include <map>
#include <memory>
#include <mutex>

#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::llh {

class SymbolAnalysis {
 public:
  static SymbolAnalysis *getInstance();
  static void deleteInstance();

  SymbolicIntOp buildSymbolInt(OpBuilder *builder, Operation *op);

  void debugPrintSymbols();

 private:
  SymbolAnalysis();
  ~SymbolAnalysis();

  SymbolAnalysis(const SymbolAnalysis &signal);
  const SymbolAnalysis &operator=(const SymbolAnalysis &signal);

 private:
  static SymbolAnalysis *instance_;
  static std::mutex mutex_;
  std::map<mlir::StringRef, Operation *> symbols_;
};

}  // namespace mlir::llh
#endif  //  INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_SYMBOLANALYSIS_H_
