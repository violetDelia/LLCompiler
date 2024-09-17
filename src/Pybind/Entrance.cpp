#include "llcompiler/Compiler/Entrance.h"

#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace llc::compiler {

extern "C" void test() {
  // do_compiler();
  std::cout << "Testing" << std::endl;
};

extern "C" void test2() {
  do_compiler();
  std::cout << "Testing" << std::endl;
};

PYBIND11_MODULE(llcompiler, llcompiler) {
  auto entrance = llcompiler.def_submodule("entrance");
  entrance.doc() = "entrance for compiler";  // optional module docstring
  entrance.def("test", &test, "A function which adds two numbers");
  entrance.def("test2", &test2, "A function which adds two numbers");
}
}  // namespace llc::compiler
