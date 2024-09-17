#include "llcompiler/Compiler/Entrance.h"

#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace llc::compiler {

PYBIND11_MODULE(llcompiler_, llcompiler_) {
  auto entrance = llcompiler_.def_submodule("entrance");
  entrance.doc() = "entrance for compiler";  // optional module docstring
  entrance.def("do_compile", &do_compile,
               "A function which adds two numbers");
}
}  // namespace llc::compiler
