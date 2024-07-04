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
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "llcompiler/llcompiler.h"
#include "onnx/common/file_utils.h"
#include "onnx/onnx-data_pb.h"

class module {};

template <class A>
class ImporterInit {
  static void init();
};

template <class A>
class Importer {
 public:
  module *import() {
    init();
    builder();
    return module_;
  }
  virtual void init();
  virtual void builder();
  virtual ~Importer() {}

  module *module_;
  ImporterInit<A> *init_imp;
};

template <class dialect>
struct Builder {
  void mlirgen();
};

int main(int argc, char **argv) {
  llc::init_compiler(argc, argv);
  return 0;
}
