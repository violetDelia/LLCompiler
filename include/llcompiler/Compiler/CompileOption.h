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

#include "llcompiler/Compiler/ToolPath.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Enums.h"
#include "mlir/IR/BuiltinOps.h"

namespace llc::compiler {

extern "C" struct CompileOptions {
  CompileOptions();

  void setLogRoot(std::string log_root);
  void setMode(std::string mode);
  void setTarget(std::string target);
  void setLogLevel(std::string log_level);
  void setPipeline(std::string pipeline);
  void setGlobalLayout(std::string global_layout);
  void setCpu(std::string cpu);
  void setMtriple(std::string mtriple);
  void displayMlirPasses(bool display);
  void displayLlvmPasses(bool display);

  std::string pipeline = "transform";             // 采用的pipeline
  llc::ModeKind mode = llc::ModeKind::inference;  // 模型运行模式
  llc::Target target = llc::Target::x86_64;       // 后端
  llc::LogLevel log_level = llc::LogLevel::debug;
  llc::GlobalLayout global_layout = llc::GlobalLayout::NCHW;

  unsigned opt_level = 3;
  uint64_t L3_cache_size = 0;
  uint64_t L2_cache_size = 0;
  uint64_t L1_cache_size = 0;
  unsigned index_bit_width;
  std::string log_root;  // 日志路径
  bool display_mlir_passes = true;
  bool display_llvm_passes = true;
  std::string mcpu;
  std::string mtriple;
  std::vector<std::string> libdirs = defaultLibDirs;
  std::vector<std::string> libs = defaultLib;
};

std::string getOptimizationLevelOption(const CompileOptions& options);
std::string getTargetArchOption(const CompileOptions& options);
std::string getCPUOption(const CompileOptions& options);
std::string getMtripleOption(const CompileOptions& options);
}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_COMPILEOPTION_H_