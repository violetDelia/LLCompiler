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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llcompiler/Compiler/Tensor.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
namespace llc::compiler {

Engine::Engine(std::unique_ptr<llvm::orc::LLJIT> engine)
    : engine(std::move(engine)) {}

void Engine::debug_info() { DINFO << engine.get(); }

struct MemerefDiscript {
  void* base;
  void* data;
  void* offset;
  void* sizes;
  void* strides;
};

int Engine::run(std::vector<Tensor*>& inputs, std::vector<Tensor*>& outs) {
  auto maybe_func = engine->lookup("main");  // 查找入口函数
  CHECK(llc::GLOBAL, maybe_func) << "count not find function!";
  auto& func = maybe_func.get();
  std::vector<void*> params;
  for (auto tensor : inputs) {
    params.push_back(static_cast<void*>(tensor->base));
    params.push_back(static_cast<void*>(tensor->data));
    params.push_back(static_cast<void*>(&tensor->offset));
    params.push_back(static_cast<void*>(tensor->size.data()));
    params.push_back(static_cast<void*>(tensor->stride.data()));
  }
  for (auto tensor : outs) {
    params.push_back(static_cast<void*>(tensor->base));
    params.push_back(static_cast<void*>(tensor->data));
    params.push_back(static_cast<void*>(&tensor->offset));
    params.push_back(static_cast<void*>(tensor->size.data()));
    params.push_back(static_cast<void*>(tensor->stride.data()));
  }
  auto run = func.toPtr<void(void**)>();  // 入口函数
  run(static_cast<void**>(params.data()));
  return 0;
}

}  // namespace llc::compiler
