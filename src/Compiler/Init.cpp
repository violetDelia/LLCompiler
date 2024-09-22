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
#include "llcompiler/Compiler/Init.h"

#include <iostream>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace llc::compiler {

void load_dialect(mlir::MLIRContext& context) {
  context.getOrLoadDialect<mlir::llh::LLHDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
}

void add_extension_and_interface(mlir::DialectRegistry& registry) {
  mlir::func::registerInlinerExtension(registry);
}

void init_logger(const logger::LoggerOption& logger_option) {
  logger::register_logger(GLOBAL, logger_option);
  logger::register_logger(UTILITY, logger_option);
  logger::register_logger(IMPORTER, logger_option);
  logger::register_logger(MLIR, logger_option);
  logger::register_logger(MLIR_PASS, logger_option);
  logger::register_logger(DEBUG, logger_option);
  INFO(GLOBAL) << "log root is: " << logger_option.path;
  INFO(GLOBAL) << "log level is: "
               << logger::log_level_to_str(logger_option.level);
}

void init_frontend(const front::FrontEndOption& front_option,
                   const logger::LoggerOption& logger_option) {
  logger::register_logger(IMPORTER, logger_option);
  INFO(GLOBAL) << "frontend type is: "
               << front::frontend_type_to_str(front_option.frontend_type);
  INFO(GLOBAL) << "input file is: " << front_option.input_file;
  INFO(GLOBAL) << "output file is: " << front_option.output_file;
  INFO(GLOBAL) << "convert onnx: " << front_option.onnx_convert;
  if (front_option.onnx_convert) {
    INFO(GLOBAL) << "convert onnx to: " << front_option.onnx_convert_version;
  }
}

}  // namespace llc::compiler
