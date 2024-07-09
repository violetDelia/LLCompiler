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
#include "llcompiler/Support/Core.h"

namespace llc {
/**********  ENUM convert  **********/
namespace logger {
const char *log_lever_to_str(const LOG_LEVER lever) {
  switch (lever) {
    case LOG_LEVER::DEBUG_:
      return "debug";
    case LOG_LEVER::INFO_:
      return "info";
    case LOG_LEVER::WARN_:
      return "warn";
    case LOG_LEVER::ERROR_:
      return "error";
    case LOG_LEVER::FATAL_:
      return "fatal";
  }
  return "unimplemented";
}
}  // namespace logger

namespace importer {
const char *importer_type_to_str(const IMPORTER_TYPE type) {
  switch (type) {
    case IMPORTER_TYPE::ONNX_FILE:
      return "onnx_file";
  }
  return "unimplemented";
}

const char *target_dialect_to_str(const TARGET_DIALECT dialect) {
  switch (dialect) {
    case TARGET_DIALECT::LLH:
      return "llh";
  }
  return "unimplemented";
}
};  // namespace importer

/**********  log module define  **********/
const char *GLOBAL = "global";
const char *IMPORTER = "importer";
}  // namespace llc
