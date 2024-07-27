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

#include <any>
#include <cstddef>
#include <memory>

#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Frontend/Core/Importer.h"
#include "llcompiler/Frontend/Core/Option.h"
#include "llcompiler/Frontend/Core/Utility.h"

namespace llc::front {

ImporterOption get_importer_option() {
  return {.filename = option::importingPath,
          .onnx_convert = option::onnxConvert,
          .onnx_convert_version = option::onnxConvertVersion,
          .frontend_type = option::frontendType};
}

}  // namespace llc::front
