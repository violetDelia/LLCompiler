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
#ifndef INCLUDE_LLCOMPILER_FRONTEND_CORE_BASE_H_
#define INCLUDE_LLCOMPILER_FRONTEND_CORE_BASE_H_
#include <cstdint>
#include <string>
namespace llc::front {
enum class FRONTEND_TYPE : int64_t { ONNX_FILE = 1, MLIR_FILE = 2 };
const char *frontend_type_to_str(const FRONTEND_TYPE type);
extern const int64_t ONNX_ADAPTED_VERSION;

struct FrontEndOption {
  std::string input_file;
  std::string output_file;
  bool onnx_convert;
  uint64_t onnx_convert_version;
  FRONTEND_TYPE frontend_type;
};

}  // namespace llc::front
#endif  // INCLUDE_LLCOMPILER_FRONTEND_CORE_BASE_H_
