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
//

#ifndef INCLUDE_LLCOMPILER_SUPPORT_ENUMS_H_
#define INCLUDE_LLCOMPILER_SUPPORT_ENUMS_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

#define LLCOMPILER_MACRO_FOR_FIX_HEAD
#include "llcompiler/Support/Eunms.h.inc"
#undef LLCOMPILER_MACRO_FOR_FIX_HEAD
#endif  // INCLUDE_LLCOMPILER_SUPPORT_ENUMS_H_

