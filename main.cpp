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
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "include/llcompiler/Dialect/LLH/IR/LLHTypes.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Importer/Utility.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnx_pb.h"

// namespace llc::importer
int main(int argc, char **argv) {
  llc::importer::ImporterOption options{
      .filename = "C:/LLCompiler/tutorials/models/mnist-12.onnx",
      .onnx_convert_version = 16,
      .importer_type = llc::importer::IMPORTER_TYPE::ONNX_FILE,
      .target_dialect = llc::importer::TARGET_DIALECT::LLH};
  mlir::MLIRContext context;
  auto import_option = llc::importer::get_importer_option();
  auto module = llc::importer::gen_mlir_from(&context, options);
  return 0;
}
