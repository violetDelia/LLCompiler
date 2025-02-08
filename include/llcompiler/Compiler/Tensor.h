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

#ifndef INCLUDE_LLCOMPILER_COMPILER_TENSOR_H_
#define INCLUDE_LLCOMPILER_COMPILER_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llc::compiler {

enum Type : size_t {
  INT8 = 0,
  INT16 = 1,
  INT32 = 2,
  INT64 = 3,
  FLOAT32 = 4,
  DOUBL64 = 5,
  INT1 = 6,
};

// 这个类只是个包装，不负责分配内存，只是记录从python传入的tensor信息。
extern "C" struct Tensor {
  Tensor();
  Tensor(size_t data_ptr, size_t base_ptr, size_t type, size_t offset,
         std::vector<size_t>& size, std::vector<size_t>& stride);
  void print();

  void* data;
  void* base;
  Type type;
  size_t offset;
  std::vector<size_t> size;
  std::vector<size_t> stride;
};

}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_TENSOR_H_

