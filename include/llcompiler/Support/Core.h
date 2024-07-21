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
/**
 * @file Core.h
 * @brief eunm define
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024  时光丶人爱
 *
 */
#ifndef INCLUDE_LLCOMPILER_SUPPORT_CORE_H_
#define INCLUDE_LLCOMPILER_SUPPORT_CORE_H_
#if __cplusplus > 201703L
#define LLC_CONSTEXPR constexpr
#else
#define LLC_CONSTEXPR
#endif  // __cplusplus > 201703L

#include <utility>
/**********  alias define  **********/
#define ALIAS_CLASS(Alias_Class, Original_Class) \
  using Alias_Class = Original_Class;
#define ALIAS_CLASS_1(Alias_Class, Original_Class) \
  template <class Arg>                             \
  using Alias_Class = Original_Class<Arg>;
#define ALIAS_FUNCTION(Alias_Func, Original_Func)                            \
  template <typename... Args>                                                \
  inline auto Alias_Func(Args &&...args) -> decltype(Original_Func(          \
                                             std::forward<Args>(args)...)) { \
    return Original_Func(std::forward<Args>(args)...);                       \
  }
namespace llc {
/**********  ENUM define  **********/
namespace logger {
enum class LOG_LEVER {
  DEBUG_ = 1,
  INFO_ = 2,
  WARN_ = 3,
  ERROR_ = 4,
  FATAL_ = 5,
};

const char *log_lever_to_str(const LOG_LEVER lever);
}  // namespace logger

namespace importer {
enum class IMPORTER_TYPE { ONNX_FILE = 1 };
const char *importer_type_to_str(const IMPORTER_TYPE type);
enum class TARGET_DIALECT { LLH = 1 };
const char *target_dialect_to_str(const TARGET_DIALECT dialect);

}  // namespace importer

/**********  log module extern  **********/
extern const char *GLOBAL;
extern const char *IMPORTER;
extern const char *LLH;
// recommended onnx version
extern const int64_t ONNX_ADAPTED_VERSION;

}  // namespace llc

#endif  // INCLUDE_LLCOMPILER_SUPPORT_CORE_H_
