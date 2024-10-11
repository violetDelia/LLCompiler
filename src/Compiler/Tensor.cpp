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
#include "llcompiler/Compiler/Tensor.h"

#include <iostream>
namespace llc::compiler {

Tensor::Tensor() {}

Tensor::Tensor(size_t data_ptr, size_t base_ptr, size_t type, size_t offset,
               std::vector<size_t>& size, std::vector<size_t>& stride)
    : offset(offset),
      size(size),
      stride(stride),
      type(static_cast<Type>(type)) {
  data = reinterpret_cast<void*>(data_ptr);
  base = reinterpret_cast<void*>(base_ptr);
}

void Tensor::print() {
  std::cout << "data: " << data << std::endl;
  std::cout << "base: " << base << std::endl;
  std::cout << "offset: " << offset << std::endl;
  std::cout << "type: " << static_cast<int64_t>(type) << std::endl;
  std::cout << "size: ";
  for (auto s : size) {
    std::cout << " " << s;
  }
  std::cout << std::endl;
  std::cout << "size: ";
  for (auto s : stride) {
    std::cout << " " << s;
  }
  std::cout << std::endl;

  float* data_ptr = reinterpret_cast<float*>(data);
  std::cout << "first data: " << data_ptr[0] << std::endl;
}

}  // namespace llc::compiler
