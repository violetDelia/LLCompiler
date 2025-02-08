

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "llcompiler/Compiler/CompileOption.h"
#include "llcompiler/Compiler/Compiler.h"
#include "llcompiler/Compiler/Execution.h"
using namespace std;
int main() {
  std::string mlir_module_file = "/home/lfr/LLCompiler/module.mlir";
  auto options = llc::compiler::CompileOptions();
  options.setLogRoot("/home/lfr/LLCompiler/ir_tree/test");
  options.setLogLevel("debug");
  options.setMode("training");
  options.setTarget("x86_64");
  options.setPipeline("transform");
  options.setCpu("tigerlake");
  options.setMtriple("x86_64-linux-gnu");
  auto compiler = llc::compiler::LLCCompiler();
  auto so_file =  compiler.generateSharedLibFromMLIRFile(mlir_module_file, options);
  auto executor = llc::compiler::Execution();
  executor.load(so_file);

  
}