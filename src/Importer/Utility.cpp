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

#include "llcompiler/Importer/OnnxImporter.h"
#include "llcompiler/Importer/Utility.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Option.h"
#include "llvm/Support/CommandLine.h"
#include "onnx/onnx_pb.h"

namespace llc::importer {
std::any get_importer_input_form_option() {
  auto importer_type = llc::option::importingType.getValue();
  switch (importer_type) {
    case importer::IMPORTER_TYPE::ONNX_FILE:
      return {option::importingPath.getValue()};
  }
  FATAL(GLOBAL) << "Unimplemented importer type: "
                << importer::importer_type_to_str(importer_type);
  return {};
}

mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_form_onnx_file(
    const std::string file) {
  ONNX_NAMESPACE::ModelProto model;
}

mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_from(
    const mlir::MLIRContext &context, const llc::importer::IMPORTER_TYPE type,
    const std::any input) {
  INFO(IMPORTER) << "---------- Begin Importing ----------";
  CHECK(IMPORTER, input.has_value()) << "input contains no value";
  INFO(IMPORTER) << "import tpye is: " << importer_type_to_str(type);
  switch (type) {
    case llc::importer::IMPORTER_TYPE::ONNX_FILE:
      auto path = std::any_cast<std::string>(input);
      INFO(IMPORTER) << "onnx file path is: " << path.c_str();

      return {};
  }
  FATAL(IMPORTER) << "Unimplemented importer type: "
                  << importer::importer_type_to_str(type);
  return {};
}
}  // namespace llc::import
