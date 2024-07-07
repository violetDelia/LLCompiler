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
#include <any>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "llcompiler/llcompiler.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llcompiler/dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/dialect/LLH/IR/LLHOps.h"
#include "llcompiler/dialect/LLH/IR/LLHTypes.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnx-data_pb.h"
#include "onnx/shape_inference/implementation.h"

namespace llc::importer {

class Importer {
 public:
  Importer(IMPORTER_TYPE type, std::any input) {}

  virtual void init() {}
  virtual void builder() {}
  virtual mlir::ModuleOp import() {}
  virtual ~Importer() {}

 private:
  // mlir::OpBuilder builder_;
  llc::importer::IMPORTER_TYPE type_;
};

template <class dialect>
struct Builder {
  void mlirgen();
};

mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_form_onnx_file(std::string file) {}

// mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_from(
//     mlir::MLIRContext &context, llc::importer::IMPORTER_TYPE type,
//     std::any input) {
//   switch (type) {
//     case llc::importer::IMPORTER_TYPE::ONNX_FILE:
//       return ;
//   }
// };
}  // namespace llc::importer
int main(int argc, char **argv) {
  llc::init_compiler(argc, argv);
  mlir::MLIRContext context;
  context.getOrLoadDialect<llc::llh::LLHDialect>();
  auto input = llc::get_importer_input_form_option();
  //  ONNX_NAMESPACE::ModelProto model;
  //  ONNX_NAMESPACE::LoadProtoFromPath(llc::option::importingPath.getValue(),
  //                                    model);
  //  ONNX_NAMESPACE::shape_inference::InferShapes(model);
  //  std::unique_ptr<ONNX_NAMESPACE::Graph> g(
  //      ONNX_NAMESPACE::ImportModelProto(model));
  //  std::cout << g->docString() << std::endl;
  //  auto nodes = g.nodes();

  // auto in =  g->inputs();
  // auto out = g->outputs();
  // std::cout<<g->name()<<std::endl;
  return 0;
}
