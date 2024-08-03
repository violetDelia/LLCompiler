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
/**
 * @file Opbuilder.h
 * @brief interface class OpBuilder that build Ops form inputs.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "onnx/common/ir.h"

#ifndef INCLUDE_LLCOMPILER_IMPORTER_OPBUILDER_H_
#define INCLUDE_LLCOMPILER_IMPORTER_OPBUILDER_H_
#define LLC_OPBUILDER_VIRTUAL_MLIRGEN(Ty)                         \
  virtual void gen_mlir(mlir::ModuleOp* module, const Ty& item) { \
    UNIMPLEMENTED(IMPORTER);                                      \
  }

#define LLC_OVERRIDE_OPBULDER_MLIRGEN(Ty) \
  void gen_mlir(mlir::ModuleOp* module, const Ty& item) final;

#define LLC_OPBULDER_MLIRGEN_IMPL(class, Ty) \
  void class ::gen_mlir(mlir::ModuleOp* module, const Ty& item)

namespace llc::importer {

class OpBuilder {
 public:
  explicit OpBuilder(mlir::MLIRContext* context);
  virtual ~OpBuilder();
  mlir::OpBuilder& builder();

  LLC_OPBUILDER_VIRTUAL_MLIRGEN(ONNX_NAMESPACE::Graph)

 protected:
  mlir::OpBuilder builder_;
};

class OpBuilderTrace {
 public:
  explicit OpBuilderTrace(OpBuilder* builder);
  virtual ~OpBuilderTrace();

  template <class Ty>
  auto gen_mlir(mlir::ModuleOp* module, const Ty& item) const;

  mlir::OpBuilder& builder() const;

 protected:
  OpBuilder* builder_;
};

template <class Ty>
auto OpBuilderTrace::gen_mlir(mlir::ModuleOp* module, const Ty& item) const {
  DEBUG(IMPORTER) << "call " << typeid(Ty).name() << " gen_mlir";
  return builder_->gen_mlir(module, item);
}

}  // namespace llc::importer
#undef LLC_OPBUILDER_VIRTUAL_MLIRGEN
#endif  // INCLUDE_LLCOMPILER_IMPORTER_OPBUILDER_H_