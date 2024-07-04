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

#include "llcompiler/utils/option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace llc {

void init_logger(std::initializer_list<std::string> modules) {
  for (auto &module : modules) {
    LLCOMPILER_INIT_LOGGER(module.c_str(), option::logRoot.getValue().data(),
                           option::logLevel.getValue())
  }
}

void init_compiler(int argc, char **argv) {
  llvm::InitLLVM compiler(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "LLCompiler: A graph compiler for ONNX models");
  init_logger({GLOBAL, IMPORTER});
  INFO(GLOBAL) << "import type is: "
               << importer::importer_type_to_str(
                      option::importingType.getValue());
  INFO(GLOBAL) << "import file is: " << option::importingPath.getValue();
}

}  // namespace llc
