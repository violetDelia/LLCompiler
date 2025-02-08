#include <string>

#include "llcompiler/Compiler/CompileOption.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"

namespace llc::compiler {
CompileOptions::CompileOptions() {}

void CompileOptions::setLogRoot(std::string log_root) {
  this->log_root = log_root;
}

void CompileOptions::setMode(std::string mode) {
  auto mode_kind = llc::symbolizeModeKind(mode);
  CHECK(llc::GLOBAL, mode_kind.has_value());
  this->mode = mode_kind.value();
}

void CompileOptions::setTarget(std::string target) {
  auto target_kind = llc::symbolizeTarget(target);
  CHECK(llc::GLOBAL, target_kind.has_value());
  this->target = target_kind.value();
}

void CompileOptions::setLogLevel(std::string log_level) {
  auto log_level_kind = llc::symbolizeLogLevel(log_level);
  CHECK(llc::GLOBAL, log_level_kind.has_value());
  this->log_level = log_level_kind.value();
};

void CompileOptions::setPipeline(std::string pipeline) {
  this->pipeline = pipeline;
};
void CompileOptions::setGlobalLayout(std::string global_layout) {
  auto global_layout_kind = llc::symbolizeGlobalLayout(global_layout);
  CHECK(llc::GLOBAL, global_layout_kind.has_value());
  this->global_layout = global_layout_kind.value();
}

void CompileOptions::setCpu(std::string cpu) { this->mcpu = cpu; }

void CompileOptions::setMtriple(std::string mtriple) {
  this->mtriple = mtriple;
}

void CompileOptions::displayMlirPasses(bool display) {
  this->display_mlir_passes = display;
}
void CompileOptions::displayLlvmPasses(bool display) {
  this->display_llvm_passes = display;
}

std::string getOptimizationLevelOption(const CompileOptions& options) {
  switch (options.opt_level) {
    case 0:
      return "-O0";
    case 1:
      return "-O1";
    case 2:
      return "-O2";
    case 3:
      return "-O3";
  }
  FATAL(llc::GLOBAL) << "Unexpected optimization level";
  return "";
}

std::string getTargetArchOption(const CompileOptions& options) {
  std::string arch_str = stringifyTarget(options.target).str();
  switch (options.target) {
    case llc::Target::x86_64:
      arch_str = "x86-64";
      break;
  }
  return "--march=" + arch_str;
}

std::string getCPUOption(const CompileOptions& options) {
  if (options.mcpu.empty()) return "";
  return "--mcpu=" + options.mcpu;
}

std::string getMtripleOption(const CompileOptions& options) {
  if (options.mtriple.empty()) return "";
  return "--mtriple=" + options.mtriple;
}
}  // namespace llc::compiler
