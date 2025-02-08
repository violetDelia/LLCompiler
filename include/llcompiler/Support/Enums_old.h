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

#ifndef INCLUDE_LLCOMPILER_SUPPORT_ENUMS_H_
#define INCLUDE_LLCOMPILER_SUPPORT_ENUMS_H_
#include <cstdint>
#include <string>

namespace llc {
enum MODE : int64_t { Training = 1, Inference = 2 };
const char *mode_to_str(const MODE level);
MODE str_to_mode(const char *);

enum TARGET : std::int64_t {
  CPU = 0,
};
const char *target_to_str(const TARGET target);
TARGET str_to_target(const char *str);

enum FRONT : std::int64_t {
  Torch = 0,
  Onnx = 1,
};
const char *front_to_str(const FRONT front);
FRONT str_to_front(const char *str);
}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_SUPPORT_ENUMS_H_
