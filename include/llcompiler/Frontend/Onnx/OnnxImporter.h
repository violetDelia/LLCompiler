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
 * @file OnnxImporter.h
 * @brief implementation of Importer that convert onnx.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */
#ifndef INCLUDE_LLCOMPILER_FRONTEND_ONNX_ONNXIMPORTER_H_
#define INCLUDE_LLCOMPILER_FRONTEND_ONNX_ONNXIMPORTER_H_
#include <cstdint>
#include <map>
#include <string>

#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Frontend/Core/Importer.h"
#include "llcompiler/Frontend/Core/Macro.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "onnx/common/ir.h"

namespace llc::front {
class OnnxImporter : public Importer {
 public:
  OnnxImporter(Builder *builder, const FrontEndOption &option);

  mlir::ModuleOp export_mlir_module() const final;

  LLC_MLIR_GEN(mlir::ModuleOp, const ONNX_NAMESPACE::ModelProto &model)
  LLC_MLIR_GEN(mlir::func::FuncOp, const ONNX_NAMESPACE::Graph &graph)
  LLC_MLIR_GEN(mlir::ShapedType, const ONNX_NAMESPACE::Value &value)
  LLC_MLIR_GEN(mlir::Type, const int32_t &elem_type)
  LLC_MLIR_GEN(
      llvm::SmallVector<mlir::Type>,
      const ONNX_NAMESPACE::ArrayRef<const ONNX_NAMESPACE::Value *> &values)
  LLC_MLIR_GEN(mlir::tosa::ConstOp, const ONNX_NAMESPACE::Tensor &weight,
               std::map<std::string, mlir::ShapedType> *weight_shape_map)
  LLC_MLIR_GEN(mlir::Operation *, const ONNX_NAMESPACE::Node &node,
               std::map<std::string, mlir::Value> *value_map)
  LLC_MLIR_GEN(mlir::Attribute, const ONNX_NAMESPACE::Node &node,
               const ONNX_NAMESPACE::BuiltinSymbol &attr_kind)

 protected:
  bool init_model_(const mlir::StringRef filename,
                   ONNX_NAMESPACE::ModelProto *model);
  bool init_model_form_json_(const mlir::StringRef &filename,
                             ONNX_NAMESPACE::ModelProto *model);
  bool init_model_form_onnx_(const mlir::StringRef &filename,
                             ONNX_NAMESPACE::ModelProto *model);
  bool check_model_legal_(const ONNX_NAMESPACE::ModelProto &model) const;
  int64_t get_model_version_(const ONNX_NAMESPACE::ModelProto &model) const;
  ONNX_NAMESPACE::ModelProto conver_model_version_to_(
      ONNX_NAMESPACE::ModelProto *model, const int64_t version);

 protected:
  ONNX_NAMESPACE::ModelProto model_;
};
}  // namespace llc::front
#endif  // INCLUDE_LLCOMPILER_FRONTEND_ONNX_ONNXIMPORTER_H_
