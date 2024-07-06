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
#include "llcompiler/support/core.h"

namespace llc {
/**********  ENUM convert  **********/
namespace logger {
const char *log_lever_to_str(LOG_LEVER lever) {
  switch (lever) {
    case LOG_LEVER::DEBUG:
      return "debug";
    case LOG_LEVER::INFO:
      return "info";
    case LOG_LEVER::WARN:
      return "warn";
    case LOG_LEVER::ERROR:
      return "error";
    case LOG_LEVER::FATAL:
      return "fatal";
  }
}
}  // namespace logger

namespace importer {
const char *importer_type_to_str(IMPORTER_TYPE type) {
  switch (type) {
    case ONNX_FILE:
      return "onnx_file";
  }
}
};  // namespace importer
/**********  log module define  **********/
const char *GLOBAL = "global";
const char *IMPORTER = "importer";
}  // namespace llc
