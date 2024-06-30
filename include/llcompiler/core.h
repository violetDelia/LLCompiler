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

#ifndef INCLUDE_LLCOMPILER_CORE_H_
#define INCLUDE_LLCOMPILER_CORE_H_
#if __cplusplus > 201703L
#define LLC_CONSTEXPR constexpr
#else
#define LLC_CONSTEXPR
#endif  // __cplusplus > 201703L
#endif  // INCLUDE_LLCOMPILER_CORE_H_
#include <memory>
#include <string>
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
/**********  external class define  **********/
ALIAS_CLASS(String, std::string)
ALIAS_CLASS_1(SharedPtr, std::shared_ptr)

}  // namespace llc

/**********  log module define  **********/
extern const char  *LLC_OPTION;
