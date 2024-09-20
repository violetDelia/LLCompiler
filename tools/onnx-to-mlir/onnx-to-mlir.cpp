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
#include "llcompiler/Compiler/Utility.h"
#include "llcompiler/Dialect/Utility/File.h"
#include "llcompiler/Frontend/Core/Option.h"
#include "llcompiler/Support/Option.h"
#define LLCOMPILER_HAS_LOG
// C:\LLCompiler\build\tools\onnx-to-mlir\onnx-to-mlir.exe
// --import-type=onnx_file
// --log-root=C:\LLCompiler\test\onnx-to-mlir\log\mnist-12 --log-lever=debug
// --input-file=C:\LLCompiler\test\models\mnist-12.onnx
// --output-file=C:\LLCompiler\test\model_ir\mnist-12.mlir
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "onnx-to-mlir");
  auto logger_option = llc::option::get_logger_option();
  auto front_option = llc::option::get_front_end_option();
  llc::compiler::init_global(logger_option);
  llc::compiler::init_frontend(front_option, logger_option);
  mlir::MLIRContext context;
  auto module = llc::compiler::gen_mlir_from(&context, front_option);
  if (front_option.output_file != "")
    llc::file::mlir_to_file(&module, front_option.output_file.c_str());
  return 0;
}
