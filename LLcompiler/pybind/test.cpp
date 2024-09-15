#include "llcompiler/Compiler/Init.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
PYBIND11_MODULE(compiler, m) { m.def("test", &llc::compiler::test); }
