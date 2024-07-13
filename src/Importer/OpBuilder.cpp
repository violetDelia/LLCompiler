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

#include "llcompiler/Importer/OpBuilder.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace llc::importer {
OpBuilder::OpBuilder(mlir::MLIRContext* context) : builder_(context) {}

OpBuilder::~OpBuilder() {}

mlir::OpBuilder& OpBuilder::build() { return builder_; }

void OpBuilder::mlirGen(mlir::ModuleOp* module, const onnx::ModelProto& model) {
  DEBUG(IMPORTER) << "gen mlirOp form onnx::ModelProto";
  mlirGen(module, model.graph());
}
void OpBuilder::mlirGen(mlir::ModuleOp* module, const onnx::GraphProto& graph) {
  DEBUG(IMPORTER) << "gen mlirOp form onnx::GraphProto";
  auto func = mlir::func::FuncOp::create(
      builder_.getUnknownLoc(), "onnx_graph",
      /*type=*/builder_.getFunctionType({}, {}), /*attrs=*/{});
  module->push_back(func);
  func.getBody().push_back(new mlir::Block);
  PRINT << graph.name();
  for (const auto& initializer : graph.initializer()) {
    PRINT << initializer.name();
  }
  for (const auto& input : graph.input()) {
    PRINT << input.name();
  }
  module->dump();
}
};  // namespace llc::importer
