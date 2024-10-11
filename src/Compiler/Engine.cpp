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

Engine::Engine(std::unique_ptr<llvm::orc::LLJIT> engine)
    : engine(std::move(engine)){};

void Engine::debug_info() { DINFO << engine.get(); }

struct Tensor_D {
  void* ptr;
  void* base;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
};

int Engine::run(std::vector<Tensor*>& inputs, std::vector<Tensor*>& outs) {
  DINFO << "in";
  auto maybe_func = engine->lookup("main");
  DINFO << "find";
  CHECK(llc::GLOBAL, maybe_func) << "count not find function!";
  auto& func = maybe_func.get();
  auto in = inputs[0];
  auto out = outs[0];
  out->print();
  auto run = func.toPtr<void(void*, void*, int, int, int, int, int, int, int,
                             int, int, void*, void*, int, int, int, int, int,
                             int, int, int, int)>();
  run(in->data, in->base, in->offset, in->size[0], in->size[1], in->size[2],
      in->size[3], in->stride[0], in->stride[1], in->stride[2], in->stride[3],
      out->data, out->base, out->offset, out->size[0], out->size[1],
      out->size[2], out->size[3], out->stride[0], out->stride[1],
      out->stride[2], out->stride[3]);

  return 0;
};

}  // namespace llc::compiler
