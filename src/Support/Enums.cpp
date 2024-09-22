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
#include "llcompiler/Support/Enums.h"

#include <cstring>
#include <exception>

#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Core.h"
namespace llc {


MODE str_to_mode(const char* str) {
  LLC_COMPARE_AND_RETURN(str, "inference", MODE::Inference)
  LLC_COMPARE_AND_RETURN(str, "training", MODE::Training)
  UNIMPLEMENTED(UTILITY) << " convert:" << str;
}

TARGET str_to_target(const char* str) {
  LLC_COMPARE_AND_RETURN(str, "cpu", TARGET::CPU)
  UNIMPLEMENTED(UTILITY) << " convert:" << str;
}

FRONT str_to_front(const char* str) {
  LLC_COMPARE_AND_RETURN(str, "torch", FRONT::Torch)
  LLC_COMPARE_AND_RETURN(str, "onnx", FRONT::Onnx)
  UNIMPLEMENTED(UTILITY) << " convert:" << str;
}
const char* front_to_str(const FRONT front) {
  switch (front) {
    case FRONT::Torch:
      return "torch";
    case FRONT::Onnx:
      return "onnx";
  }
  UNIMPLEMENTED(UTILITY);
}

const char* mode_to_str(const MODE mode) {
  switch (mode) {
    case MODE::Inference:
      return "inference";
    case MODE::Training:
      return "training";
  }
  UNIMPLEMENTED(UTILITY);
}
const char* target_to_str(const TARGET target) {
  switch (target) {
    case TARGET::CPU:
      return "cpu";
  }
  UNIMPLEMENTED(UTILITY);
}
}  // namespace llc
