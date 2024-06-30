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
 * @file logger.h
 * @brief 日志宏以及检查宏
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024  时光丶人爱
 *
 */
#ifndef INCLUDE_LLCOMPILER_UTILS_LOGGER_H_
#define INCLUDE_LLCOMPILER_UTILS_LOGGER_H_
#include <string>

#include "llcompiler/core.h"


namespace llc::logger {
class Logger;
class LoggerStream;

enum LOG_LEVER {
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
};

void register_logger(const char *module, const char *root_path,
                     const LOG_LEVER lever);

class LoggerStream {
 public:
  LLC_CONSTEXPR LoggerStream(Logger *log);
  LLC_CONSTEXPR LoggerStream &operator<<(const char *message);
  LLC_CONSTEXPR LoggerStream &operator<<(const std::string &str);
  LLC_CONSTEXPR LoggerStream &operator<<(const int value);
  LLC_CONSTEXPR LoggerStream &operator<<(const double value);
  LLC_CONSTEXPR virtual ~LoggerStream();

 protected:
  std::string message_;
  Logger *logger_;
};

class Logger {
 public:
  LLC_CONSTEXPR Logger(const char *module, LOG_LEVER level);
  LLC_CONSTEXPR LoggerStream stream();
  LLC_CONSTEXPR void info(const char *message);
  LLC_CONSTEXPR virtual ~Logger();

 protected:
  const char *module_;
  LOG_LEVER level_;
};

class NullStream {
 public:
  template <class Ty>
  LLC_CONSTEXPR NullStream &operator<<(Ty val);
};

}  // namespace llc::logger
#ifdef LLCOMPILER_HAS_LOG
#define LLCOMPILER_INIT_LOGGER(module, root, lever) \
  llc::logger::register_logger(module, root, lever);
#define LOG(module, lever) llc::logger::Logger(module, lever).stream()
#define CHECK_LOG(module, condition, lever) \
  if (condition) {                          \
    LOG(module, lever)                      \
  }
#else
#define LLCOMPILER_INIT_LOGGER(module, root, lever)
#define LOG(module, lever) llc::logger::NullStream()
#define CHECK_LOG(module, condition, lever) llc::logger::NullStream()
#endif  // LLCOMPILER_HAS_LOG

#define DEBUG(module) LOG(module, llc::logger::DEBUG)
#define INFO(module) LOG(module, llc::logger::INFO)
#define WARN(module) LOG(module, llc::logger::WARN)
#define ERROR(module) LOG(module, llc::logger::ERROR)
#define FATAL(module) LOG(module, llc::logger::FATAL)

#define CHECK(module, condition) \
  CHECK_LOG(module, condition, llc::logger::ERROR)
#define CHECK_EQ(module, val1, val2) CHECK(module, val1 == val2)
#define CHECK_NE(module, val1, val2) CHECK(module, val1 != val2)
#define CHECK_LT(module, val1, val2) CHECK(module, val1 < val2)
#define CHECK_LE(module, val1, val2) CHECK(module, val1 <= val2)
#define CHECK_GT(module, val1, val2) CHECK(module, val1 > val2)
#define CHECK_GE(module, val1, val2) CHECK(module, val1 >= val2)

#define DCHECK(module, condition) \
  CHECK_LOG(module, condition, llc::logger::DEBUG)
#define DCHECK_EQ(module, val1, val2) DCHECK(module, val1 == val2)
#define DCHECK_NE(module, val1, val2) DCHECK(module, val1 != val2)
#define DCHECK_LT(module, val1, val2) DCHECK(module, val1 < val2)
#define DCHECK_LE(module, val1, val2) DCHECK(module, val1 <= val2)
#define DCHECK_GT(module, val1, val2) DCHECK(module, val1 > val2)
#define DCHECK_GE(module, val1, val2) DCHECK(module, val1 >= val2)

#define LOG_EQ(module, val1, val2, lever) CHECK_LOG(module, val1 == val2, lever)
#define LOG_NE(module, val1, val2, lever) CHECK_LOG(module, val1 != val2, lever)
#define LOG_LT(module, val1, val2, lever) CHECK_LOG(module, val1 < val2, lever)
#define LOG_LE(module, val1, val2, lever) CHECK_LOG(module, val1 <= val2, lever)
#define LOG_GT(module, val1, val2, lever) CHECK_LOG(module, val1 > val2, lever)
#define LOG_GE(module, val1, val2, lever) CHECK_LOG(module, val1 >= val2, lever)

#define TYPE(val) typeid(val).name()

#endif  // INCLUDE_LLCOMPILER_UTILS_LOGGER_H_
