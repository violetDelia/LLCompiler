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

#include <filesystem>

#include "llcompiler/Compiler/Utility.h"
#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Frontend/Onnx/OnnxBuilder.h"
#include "llcompiler/Frontend/Onnx/OnnxImporter.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace llc::compiler {

mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_from(
    mlir::MLIRContext *context, const front::FrontEndOption &option) {
  INFO(IMPORTER) << "---------- Begin Importing ----------";
  INFO(IMPORTER) << "import tpye is: "
                 << frontend_type_to_str(option.frontend_type);

  switch (option.frontend_type) {
    case llc::front::FRONTEND_TYPE::ONNX_FILE: {
      INFO(IMPORTER) << "onnx file path is: " << option.input_file.c_str();
      auto builder = front::OnnxBuilder(context);
      return front::OnnxImporter(&builder, option).export_mlir_module();
    }
    default:
      FATAL(IMPORTER) << "Unimplemented importer type: "
                      << frontend_type_to_str(option.frontend_type);
      return {};
  }
}

}  // namespace llc::compiler
