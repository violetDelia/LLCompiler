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
#include "llcompiler/Compiler/Entrance.h"

#include <iostream>

#include "llcompiler/Pipeline/BasicPipeline.h"
#include "pybind11/pybind11.h"
namespace llc::compiler {

PYBIND11_MODULE(llcompiler_, llcompiler_) {
  auto entrance = llcompiler_.def_submodule("entrance");
  entrance.doc() = "entrance for compiler";  // optional module docstring

  pybind11::class_<llc::compiler::CompilerOptions>(entrance, "CompilerOptions")
      .def(pybind11::init<std::string, std::string, bool, unsigned, std::string,
                          std::string, std::string>())
      .def_readwrite("mode", &CompilerOptions::mode)
      .def_readwrite("target", &CompilerOptions::target)
      .def_readwrite("symbol_infer", &CompilerOptions::symbol_infer)
      .def_readwrite("index_bits", &CompilerOptions::index_bit_width)
      .def_readwrite("ir_tree_dir", &CompilerOptions::ir_tree_dir)
      .def_readwrite("log_root", &CompilerOptions::log_root)
      .def_readwrite("log_level", &CompilerOptions::log_level);

  entrance.def("do_compile", &do_compile, "");
}

}  // namespace llc::compiler
