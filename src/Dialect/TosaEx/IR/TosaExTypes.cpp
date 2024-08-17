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

#include "llcompiler/Dialect/TosaExtension/IR/TosaExDialect.h"
#include "llcompiler/Dialect/TosaExtension/IR/TosaExTypes.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define FIX_HEADER
#include "llcompiler/Dialect/TosaExtension/IR/TosaExEunms.cpp.inc"
#undef FIX_HEADER
#define GET_ATTRDEF_CLASSES
#include "llcompiler/Dialect/TosaExtension/IR/TosaExAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/TosaExtension/IR/TosaExTypes.cpp.inc"
