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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

#ifndef INCLUDE_LLCOMPILER_IMPORTER_OPBUILDER_H_
#define INCLUDE_LLCOMPILER_IMPORTER_OPBUILDER_H_
#define DEFINE_OPBUILDER_VIRTUAL_MLIRGEN(Ty)                      \
  virtual void gen_mlir(mlir::ModuleOp* module, const Ty& item) { \
    UNIMPLEMENTED(IMPORTER);                                      \
  }

#define LLCOMPILER_OVERRIDE_OPBULDER_MLIRGEN(Ty) \
  void gen_mlir(mlir::ModuleOp* module, const Ty& item) final;

#define LLCOMPILER_OPBULDER_MLIRGEN_IMPL(class, Ty) \
  void class ::gen_mlir(mlir::ModuleOp* module, const Ty& item)
namespace llc::importer {

class OpBuilder {
 public:
  explicit OpBuilder(mlir::MLIRContext* context);
  virtual ~OpBuilder();
  mlir::OpBuilder& build();
  llh::IntType get_int(unsigned width = 32,
                       llh::SIGNED_TAG tag = llh::SIGNED_TAG::UNSIGNED) {
    return builder_.getType<llh::IntType>(width, tag);
  }
  // mlir::FloatType get_float(unsigned width = 32) {
  //   return builder_.getType<mlir::FloatType>(width);
  // }
  // mlir::ComplexType get_tensor() {
  //   //return builder_.getType<mlir::TensorType>()
  // };

  DEFINE_OPBUILDER_VIRTUAL_MLIRGEN(onnx::ModelProto)
  DEFINE_OPBUILDER_VIRTUAL_MLIRGEN(onnx::GraphProto)
  DEFINE_OPBUILDER_VIRTUAL_MLIRGEN(onnx::Graph)

 protected:
  mlir::OpBuilder builder_;
};

class OpBuilderTrace {
 public:
  explicit OpBuilderTrace(OpBuilder* builder);
  virtual ~OpBuilderTrace();

  template <class Ty>
  void gen_mlir(mlir::ModuleOp* module, const Ty& item) const;

  OpBuilder& build() const;

 protected:
  OpBuilder* builder_;
};

template <class Ty>
void OpBuilderTrace::gen_mlir(mlir::ModuleOp* module, const Ty& item) const {
  DEBUG(IMPORTER) << "call " << typeid(Ty).name() << " gen_mlir";
  builder_->gen_mlir(module, item);
}

}  // namespace llc::importer
#undef DEFINE_OPBUILDER_VIRTUAL_MLIRGEN
#endif  // INCLUDE_LLCOMPILER_IMPORTER_OPBUILDER_H_
