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

#ifndef INCLUDE_LLCOMPILER_COMPILER_COMMAND_H_
#define INCLUDE_LLCOMPILER_COMPILER_COMMAND_H_
#include <optional>
#include <string>
#include <vector>

#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
namespace llc::compiler {

std::string getToolPath(const std::string &tool);

struct Command {
  std::string _path;
  std::vector<std::string> _args;

  explicit Command(std::string exe_path);

  Command &appendStr(const std::string &arg);
  Command &appendStrOpt(const std::optional<std::string> &arg);
  Command &appendList(const std::vector<std::string> &args);
  Command &resetArgs();
  void exec(std::string wdir = "") const;
};

}  // namespace llc::compiler

#endif  // INCLUDE_LLCOMPILER_COMPILER_COMMAND_H_
