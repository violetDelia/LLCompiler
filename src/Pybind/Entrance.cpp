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
#include <pybind11/buffer_info.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "llcompiler/Compiler/Command.h"
#include "llcompiler/Compiler/Compiler.h"
#include "llcompiler/Compiler/Execution.h"
#include "llcompiler/Compiler/Tensor.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "mlir-c/ExecutionEngine.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;
namespace llc::compiler {

namespace {

size_t get_itemsize(llc::compiler::Type type) {
  switch (type) {
    case Type::FLOAT32:
      return sizeof(float);
    case Type::INT64:
      return sizeof(int64_t);
  }
  UNIMPLEMENTED(llc::UTILITY);
}

std::string get_format(llc::compiler::Type type) {
  switch (type) {
    case Type::FLOAT32:
      return pybind11::format_descriptor<float>::format();
    case Type::INT64:
      return pybind11::format_descriptor<int64_t>::format();
  }
  UNIMPLEMENTED(llc::UTILITY);
}

std::vector<size_t> get_stride_in(Tensor *tensor) {
  auto item_size = get_itemsize(tensor->type);
  std::vector<size_t> stride_in;
  for (auto stride : tensor->stride) {
    stride_in.push_back(stride * item_size);
  }
  return stride_in;
}

}  // namespace

PYBIND11_MODULE(llcompiler_, llcompiler_) {
  auto tensor = llcompiler_.def_submodule("tensor");
  pybind11::class_<Tensor>(tensor, "Tensor", py::buffer_protocol())
      .def_readwrite("data", &Tensor::data)
      .def(pybind11::init<size_t, size_t, size_t, size_t, std::vector<size_t> &,
                          std::vector<size_t> &>())
      .def_readwrite("data", &Tensor::data)
      .def_readwrite("base", &Tensor::base)
      .def_readwrite("offset", &Tensor::offset)
      .def_readwrite("size", &Tensor::size)
      .def_readwrite("stride", &Tensor::stride)
      .def("print", &Tensor::print)
      .def_buffer([](Tensor &self) -> py::buffer_info {
        return py::buffer_info(self.base, get_itemsize(self.type),
                               get_format(self.type), self.size.size(),
                               self.size, get_stride_in(&self));
      })
      .def("to_numpy", [](Tensor *self) {
        auto bufer = py::buffer_info(self->base, get_itemsize(self->type),
                                     get_format(self->type), self->size.size(),
                                     self->size, get_stride_in(self));
        return py::array(bufer);
      });

  auto compiler = llcompiler_.def_submodule("compiler");

  pybind11::class_<llc::compiler::LLCCompiler>(compiler, "Compiler")
      .def(pybind11::init<>())
      .def("compile_mlir_to_shared_lib",
           &LLCCompiler::generateSharedLibFromMLIRStr);

  pybind11::class_<llc::compiler::CompileOptions>(compiler, "CompileOptions")
      .def(pybind11::init<>())
      .def("set_log_root", &CompileOptions::setLogRoot)
      .def("set_mode", &CompileOptions::setMode)
      .def("set_target", &CompileOptions::setTarget)
      .def("set_log_level", &CompileOptions::setLogLevel)
      .def("set_pipeline", &CompileOptions::setPipeline)
      .def("set_global_layout", &CompileOptions::setGlobalLayout)
      .def("set_cpu", &CompileOptions::setCpu)
      .def("set_mtriple", &CompileOptions::setMtriple)
      .def("display_llvm_passes", &CompileOptions::displayLlvmPasses)
      .def("display_mlir_passes", &CompileOptions::displayMlirPasses);

  auto executor = llcompiler_.def_submodule("executor");
  executor.def("get_tool", &getToolPath);
  pybind11::class_<llc::compiler::Execution>(executor, "Execution")
      .def(pybind11::init<>())
      .def("load", &Execution::load)
      .def("run", &Execution::run)
      .def("run_with_symbols", &Execution::run_with_symbols);
}

}  // namespace llc::compiler
