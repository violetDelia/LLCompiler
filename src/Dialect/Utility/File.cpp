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
#include "llcompiler/Dialect/Utility/File.h"

#include <filesystem>

#include "llcompiler/Support/Logger.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"

namespace llc::file {
void mlir_to_file(mlir::OwningOpRef<mlir::ModuleOp>* module, const char* file) {
  std::error_code error_code;
  auto file_dir = std::filesystem::path(file).parent_path();
  if (!std::filesystem::exists(file_dir)) {
    std::filesystem::create_directory(file_dir);
    INFO(llc::GLOBAL) << "create directory " << file_dir;
  }
  llvm::raw_fd_stream file_stream(file, error_code);
  (*module)->print(file_stream);
  INFO(GLOBAL) << "module convert to file: " << file;
}

void file_to_mlir_module(mlir::MLIRContext& context,
                         mlir::OwningOpRef<mlir::ModuleOp>& module,
                         const char* file) {
  // Handle '.toy' input to the compiler.

  if (!llvm::StringRef(file).ends_with(".mlir")) {
    UNIMPLEMENTED(UTILITY);
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(file);
  if (std::error_code ec = fileOrErr.getError()) {
    FATAL(UTILITY) << "could not open input file: " << file;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    FATAL(UTILITY) << "error can't load file " << file << "\n";
  }
}

void str_to_mlir_module(mlir::MLIRContext& context,
                        mlir::OwningOpRef<mlir::ModuleOp>& module,
                        const char* str) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getMemBuffer(str, "xdsl_module");
  if (std::error_code ec = fileOrErr.getError()) {
    FATAL(UTILITY) << "load xdsl module fatal error!";
    return;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    FATAL(UTILITY) << "parse xdsl module fatal error!";
    return;
  }
  return;
}

}  // namespace llc::file
