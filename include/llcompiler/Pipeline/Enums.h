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

#ifndef INCLUDE_LLCOMPILER_PIPELINE_ENUMS_H_
#define INCLUDE_LLCOMPILER_PIPELINE_ENUMS_H_
#include <cstdint>
namespace llc::pipleline {
enum RUN_MODE : std::int64_t {
  INFERENCE = 0,
  TRAINING = 1,
};
const char* run_mode_to_str(const RUN_MODE mode);
enum TARGET : std::int64_t {
  LLVM = 0,
  SPIRV = 1,
};
const char* target_to_str(const TARGET target);
}  // namespace llc::pipleline
#endif  // INCLUDE_LLCOMPILER_PIPELINE_ENUMS_H_
