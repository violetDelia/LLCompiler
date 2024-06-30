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
#ifndef LLCOMPILER_HAS_LOG
#define LLCOMPILER_HAS_LOG
#endif
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "llcompiler/llcompiler.h"
#include "llcompiler/utils/logger.h"

// void force_hidden_options(
//     llvm::ArrayRef<const llvm::cl::OptionCategory *> categories) {
//   for (const auto &category : categories) {
//     category->
//   }
// }

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  LLCOMPILER_INIT_LOGGER(LLC_OPTION, LLC_logRoot.getValue().data(),
                         LLC_logLevel.getValue());
  return 0;
}
