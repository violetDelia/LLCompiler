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

#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h"
#include "llcompiler/Importer/LLHOpBuilder.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

namespace llc::importer {
LLHOpBuilder::LLHOpBuilder(mlir::MLIRContext* context) : OpBuilder(context) {
  context->getOrLoadDialect<llc::llh::LLHDialect>();
}
// void LLHOpBuilder::mlirGen(mlir::ModuleOp* module,
//                            const onnx::ModelProto& model) {
//   DEBUG(IMPORTER) << "gen mlirOp form onnx::ModelProto";
//   print_info << "doc_string: " << model.doc_string();
//   print_info << "domain: " << model.domain();
//   print_info << "ir_version: " << model.ir_version();
//   print_info << "model_version: " << model.model_version();
//   print_info << "opset_size: " << model.opset_import().size();
//   print_info << "producer_name: " << model.producer_name();
//   print_info << "producer_version: " << model.producer_version();
//   mlirGen(module, model.graph());
// }
// void LLHOpBuilder::mlirGen(mlir::ModuleOp* module,
//                            const onnx::GraphProto& graph) {
//   DEBUG(IMPORTER) << "gen mlirOp form onnx::GraphProto";
//   print_info << "doc_string: " << graph.doc_string();
//   for (const auto& initializer : graph.initializer()) {
//     print_info << "initializer: " << initializer.name();
//     print_info << "data_type: " << initializer.data_type();
//     print_info << "dims: ";
//     for (auto& d : initializer.dims()) {
//       std::cout << " " << d;
//     }
//     std::cout << std::endl;
//     print_info << "raw_data: " << initializer.raw_data().size();
//     // TODO: Import initializer data to MLIR.
//   }
//   auto func = mlir::func::FuncOp::create(
//       builder_.getUnknownLoc(), "onnx_graph",
//       /*type=*/builder_.getFunctionType({}, {}), /*attrs=*/{});
//   module->push_back(func);
//   func.getBody().push_back(new mlir::Block);

//   for (const auto& input : graph.input()) {
//   }
//   module->dump();
// }

LLCOMPILER_OPBULDER_MLIRGEN_IMPL(LLHOpBuilder, onnx::Graph) {
  auto input = item.inputs();
  for (auto val : input) {
    print_info << val->node();
  }
  auto func =
      mlir::func::FuncOp::create(builder_.getUnknownLoc(), item.name(),
                                 builder_.getFunctionType({get_int()}, {}),
                                 /*attrs=*/{});
  // auto int_type =get_int() ;
  module->push_back(func);
}

};  // namespace llc::importer
