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

/**
 * @file Init.h
 * @brief initializing compiler
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */

#ifndef INCLUDE_LLCOMPILER_COMPILER_ENGINE_H_
#define INCLUDE_LLCOMPILER_COMPILER_ENGINE_H_
#include <cstddef>
#include <string>
#include <vector>

#include "llcompiler/Compiler/Tensor.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
namespace llc::compiler {

extern "C" struct Engine {
  Engine(llvm::orc::LLJIT* engine);

  void debug_info();

  std::vector<Tensor*> run(std::vector<Tensor*>& inputs);

  llvm::orc::LLJIT* engine;
};

}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_ENGINE_H_
