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

#ifndef INCLUDE_LLCOMPILER_COMPILER_COMPILEOPTION_H_
#define INCLUDE_LLCOMPILER_COMPILER_COMPILEOPTION_H_
#include <cstddef>
#include <string>
#include <vector>

#include "llcompiler/Compiler/Engine.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "mlir/IR/BuiltinOps.h"

namespace llc::compiler {

extern "C" struct CompilerOptions {
  CompilerOptions();

  std::string pipeline;  // 采用的pipeline
  std::string mode;      // 模型运行模式
  std::string target;    // 后端
  bool symbol_infer;
  unsigned opt_level;
  uint64_t L3_cache_size;
  uint64_t L2_cache_size;
  uint64_t L1_cache_size;
  unsigned index_bit_width;
  std::string target_layout;  // 数据布局
  std::string log_root;       // 日志路径
  std::string log_level;      // 日志等级
  bool log_llvm;              // 输出bitcode
};

}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_COMPILEOPTION_H_