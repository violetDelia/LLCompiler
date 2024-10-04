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

#ifndef INCLUDE_LLCOMPILER_COMPILER_ENTRANCE_H_
#define INCLUDE_LLCOMPILER_COMPILER_ENTRANCE_H_
#include <string>
namespace llc::compiler {

extern "C" struct CompilerOptions {
  CompilerOptions(std::string mode, std::string target, bool symbol_infer,unsigned index_bits,
                  std::string ir_tree_dir, std::string log_root,
                  std::string log_level);
  std::string mode;
  std::string target;
  bool symbol_infer;
  unsigned index_bit_width;
  std::string ir_tree_dir;
  std::string log_root;
  std::string log_level;
};

extern "C" void do_compile(const char* xdsl_module,
                           CompilerOptions compiler_options);

}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_ENTRANCE_H_
