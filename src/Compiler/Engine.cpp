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

#include "llcompiler/Support/Logger.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace llc::compiler {

Engine::Engine(llvm::orc::LLJIT* engine) : engine(engine){};

void Engine::debug_info() { DINFO << engine; }

std::vector<Tensor*> Engine::run(std::vector<Tensor*>& inputs){
  for(auto t : inputs){
    t->print();
  }
  return inputs;

};

}  // namespace llc::compiler
