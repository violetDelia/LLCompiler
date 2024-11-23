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
#include <cstdint>

#include "Frontend/Core/Base.h"

namespace llc::front {
const int64_t ONNX_ADAPTED_VERSION = 22;

const char *frontend_type_to_str(const FRONTEND_TYPE type) {
  switch (type) {
    case FRONTEND_TYPE::ONNX_FILE:
      return "onnx_file";
    case FRONTEND_TYPE::MLIR_FILE:
      return "mlir_file";
  }
  return "unimplemented";
}

}  // namespace llc::front
