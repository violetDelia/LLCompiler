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
#include <initializer_list>
#include <string>

#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/support/Option.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"

namespace llc {

void init_logger_(std::initializer_list<std::string> modules) {
  for (auto &module : modules) {
    LLCOMPILER_INIT_LOGGER(module.c_str(), option::logRoot.getValue().data(),
                           option::logLevel.getValue())
  }
}

void init_compiler(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "LLCompiler: A graph compiler for ONNX models");
  init_logger_({GLOBAL, IMPORTER});
  INFO(GLOBAL) << "import type is: "
               << importer::importer_type_to_str(
                      option::importingType.getValue());
  INFO(GLOBAL) << "import file is: " << option::importingPath.getValue();
  INFO(GLOBAL) << "importer target dialect is: "
               << importer::target_dialect_to_str(
                      option::importintDialect.getValue());
}

}  // namespace llc
