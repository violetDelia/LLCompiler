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

#ifndef INCLUDE_LLCOMPILER_COMPILER_EXECUTION_H_
#define INCLUDE_LLCOMPILER_COMPILER_EXECUTION_H_
#include <string>

#include "llcompiler/Compiler/Tensor.h"
#include "llvm/Support/DynamicLibrary.h"
namespace llc::compiler {

using entryPointFuncType = void (*)(void**);

class Execution {
 public:
  explicit Execution();
  ~Execution();

  void run(std::vector<Tensor*>& inputs, std::vector<Tensor*>& outs);

  void run_with_symbols(std::vector<int64_t>& symbols, std::vector<Tensor*>& inputs,
           std::vector<Tensor*>& outs);

  void load(std::string shared_lib_path);

 protected:
  bool is_initialized_ = false;
  llvm::sys::DynamicLibrary shared_lib_handle_;
  entryPointFuncType entry_point_func_;
};

}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_EXECUTION_H_
