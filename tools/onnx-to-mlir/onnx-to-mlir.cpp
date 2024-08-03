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

#include "iostream"
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Compiler/Utility.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Frontend/Core/Option.h"
#include "llcompiler/Frontend/Core/Utility.h"
#include "llcompiler/Support/Core.h"
#define LLCOMPILER_HAS_LOG
// namespace llc::importer
int main(int argc, char **argv) {
  llc::compiler::init_compiler(argc, argv);
  mlir::MLIRContext context;
  auto front_option = llc::option::get_front_end_option();
  auto module = llc::compiler::gen_mlir_from(&context, front_option);
  std::error_code error_code;
  llvm::raw_fd_stream file_stream(front_option.output_file, error_code);
  module->print(file_stream);
  llvm::outs() << "onnx convert to mgb dialect done.\n";
  return 0;
}
