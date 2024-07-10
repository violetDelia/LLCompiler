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
#include <cstddef>

#include "llcompiler/Importer/LLHOpBuilder.h"
#include "llcompiler/Importer/OnnxImporter.h"
#include "llcompiler/Importer/Utility.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Option.h"
#include "llvm/Support/CommandLine.h"

namespace llc::importer {
std::any get_importer_input_form_option() {
  auto importer_type = llc::option::importingType.getValue();
  switch (importer_type) {
    case importer::IMPORTER_TYPE::ONNX_FILE:
      return {option::importingPath.getValue()};
    default:
      FATAL(GLOBAL) << "Unimplemented importer type: "
                    << importer::importer_type_to_str(importer_type);
      return {};
  }
}

mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_from_to(
    mlir::MLIRContext *context, const llc::importer::IMPORTER_TYPE type,
    const std::any input, const TARGET_DIALECT target) {
  INFO(IMPORTER) << "---------- Begin Importing ----------";
  CHECK(IMPORTER, input.has_value()) << "input contains no value";
  INFO(IMPORTER) << "import tpye is: " << importer_type_to_str(type);
  INFO(IMPORTER) << "target dialect is: " << target_dialect_to_str(target);
  OpBuilder *builder_pointer{};
  switch (target) {
    case TARGET_DIALECT::LLH: {
      auto builder = LLHOpBuilder(context);
      builder_pointer = &builder;
      DEBUG(IMPORTER) << "build LLHOpBuilder";
      break;
    }
    default:
      FATAL(IMPORTER) << "Unsupported to covert dialect: "
                      << importer::target_dialect_to_str(target);
  }
  switch (type) {
    case llc::importer::IMPORTER_TYPE::ONNX_FILE: {
      auto path = std::any_cast<std::string>(input);
      INFO(IMPORTER) << "onnx file path is: " << path.c_str();
      return OnnxImporter(context, builder_pointer, path,
                          option::onnxConvertVersion)
          .export_mlir_module();
    }
    default:
      FATAL(IMPORTER) << "Unimplemented importer type: "
                      << importer::importer_type_to_str(type);
      return {};
  }
}
}  // namespace llc::importer
