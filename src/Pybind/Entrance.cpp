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

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>

#include <iostream>

#include "llcompiler/Pipeline/BasicPipeline.h"
#include "mlir-c/ExecutionEngine.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;
namespace llc::compiler {

namespace {}  // namespace

PYBIND11_MODULE(llcompiler_, llcompiler_) {
  auto entrance = llcompiler_.def_submodule("entrance");
  entrance.doc() = "entrance for compiler";  // optional module docstring

  pybind11::class_<mlir::ExecutionEngine>(entrance, "ExecutionEngine")
      .def(pybind11::init<bool, bool, bool>());

  pybind11::class_<Engine>(entrance, "EngineInternel")
      .def(pybind11::init<mlir::ExecutionEngine*>(),
           py::arg("execution_engine_ptr"))
      .def_readonly("engine", &Engine::engine)
      .def("debug_info", &Engine::debug_info);

  pybind11::class_<llc::compiler::CompilerOptions>(entrance, "CompilerOptions")
      .def(pybind11::init<>())
      .def_readwrite("mode", &CompilerOptions::mode)
      .def_readwrite("target", &CompilerOptions::target)
      .def_readwrite("symbol_infer", &CompilerOptions::symbol_infer)
      .def_readwrite("opt_level", &CompilerOptions::opt_level)
      .def_readwrite("L3_cache_size", &CompilerOptions::L3_cache_size)
      .def_readwrite("L2_cache_size", &CompilerOptions::L2_cache_size)
      .def_readwrite("L1_cache_size", &CompilerOptions::L1_cache_size)
      .def_readwrite("index_bit_width", &CompilerOptions::index_bit_width)
      .def_readwrite("ir_tree_dir", &CompilerOptions::ir_tree_dir)
      .def_readwrite("log_root", &CompilerOptions::log_root)
      .def_readwrite("log_level", &CompilerOptions::log_level)
      .def_readwrite("log_llvm", &CompilerOptions::log_llvm);

  entrance.def("do_compile", &do_compile);
}

}  // namespace llc::compiler
