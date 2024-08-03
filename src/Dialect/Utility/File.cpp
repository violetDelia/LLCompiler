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

#include "llcompiler/Dialect/Utility/File.h"
#include "llcompiler/Support/Logger.h"

namespace llc::file {
void mlir_to_file(mlir::ModuleOp* module, const char* file) {
  std::error_code error_code;
  auto file_dir = std::filesystem::path(file).parent_path();
  if (!std::filesystem::exists(file_dir)) {
    std::filesystem::create_directory(file_dir);
    INFO(GLOBAL) << "create directory " << file_dir;
  }
  llvm::raw_fd_stream file_stream(file, error_code);
  module->print(file_stream);
  INFO(GLOBAL) << "module convert to file: " << file;
}

}  // namespace llc::file