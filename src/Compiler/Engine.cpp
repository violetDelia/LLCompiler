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
#include "llcompiler/Compiler/Engine.h"

#include <cstdint>

#include "llcompiler/Compiler/Tensor.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
namespace llc::compiler {

Engine::Engine(llvm::orc::LLJIT* engine) : engine(engine){};

void Engine::debug_info() { DINFO << engine; }

struct Tensor_D {
  void* ptr;
  void* base;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
};

std::vector<Tensor*> Engine::run(std::vector<Tensor*>& inputs) {
  auto maybe_func = engine->lookup("main");
  CHECK(llc::GLOBAL, maybe_func) << "count not find function!";
  auto& func = maybe_func.get();
  auto in = inputs[0];
  in->print();
  auto run = func.toPtr<Tensor_D(void*, void*, int, int, int, int, int, int,
                                 int, int, int)>();
  auto c = run(in->data, in->base, in->offset, in->size[0], in->size[1],
               in->size[2], in->size[3], in->stride[0], in->stride[1],
               in->stride[2], in->stride[3]);
  DINFO << c.ptr;
  DINFO << c.base;
  DINFO << c.sizes[0];
  DINFO << c.sizes[1];
  DINFO << c.sizes[2];
  DINFO << c.strides[0];
  DINFO << c.strides[1];
  DINFO << c.strides[2];
  DINFO << static_cast<float*>(c.base)[0];
  auto res = new Tensor();
  res->data = c.ptr;
  res->base = c.base;
  res->offset = c.offset;
  res->type = Type::FLOAT32;
  for (int i = 0; i < 4; i++) {
    res->size.push_back(c.sizes[i]);
    res->stride.push_back(c.strides[i]);
  }
  DINFO << c.strides[2];
  std::vector<Tensor*> out;
  out.push_back(res);
  return out;
};

}  // namespace llc::compiler
