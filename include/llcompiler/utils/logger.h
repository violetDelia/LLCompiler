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

void register_logger(const char *module, const char *root_path,
                     const LOG_LEVER lever);

class Logger {
 public:
  LLC_CONSTEXPR Logger(const char *module, LOG_LEVER level);
  LLC_CONSTEXPR virtual ~Logger();
  LLC_CONSTEXPR LoggerStream stream(const bool emit_message);
  void info(const char *message);

 protected:
  const char *module_;
  const LOG_LEVER level_;
};

class NullStream {
 public:
  template <class Ty>
  LLC_CONSTEXPR NullStream &operator<<(Ty val);
};

class LoggerStream {
 public:
  LLC_CONSTEXPR LoggerStream(Logger *log, const bool emit_message);
  LLC_CONSTEXPR LoggerStream &operator<<(const char *message);
  LLC_CONSTEXPR LoggerStream &operator<<(const std::string &str);
  LLC_CONSTEXPR LoggerStream &operator<<(const int value);
  LLC_CONSTEXPR LoggerStream &operator<<(const double value);
  LLC_CONSTEXPR virtual ~LoggerStream();

 protected:
  std::string message_;
  Logger *logger_;
  const bool emit_message_;
};

}  // namespace llc::logger
#ifdef LLCOMPILER_HAS_LOG
#define LLCOMPILER_INIT_LOGGER(module, root, lever) \
  llc::logger::register_logger(module, root, lever);
#define LLCOMPILER_LOG(module, lever) \
  llc::logger::Logger(module, lever).stream(true)
#define LLCOMPILER_CHECK_LOG(module, condition, lever) \
  llc::logger::Logger(module, lever).stream(condition)

#else
#define LLCOMPILER_INIT_LOGGER(module, root, lever)
#define LLCOMPILER_LOG(module, lever) llc::logger::NullStream()
#define LLCOMPILER_CHECK_LOG(module, condition, lever) llc::logger::NullStream()
#endif  // LLCOMPILER_HAS_LOG

#define DEBUG(module) LLCOMPILER_LOG(module, llc::DEBUG)
#define INFO(module) LLCOMPILER_LOG(module, llc::INFO)
#define WARN(module) LLCOMPILER_LOG(module, llc::WARN)
#define ERROR(module) LLCOMPILER_LOG(module, llc::ERROR)
#define FATAL(module) LLCOMPILER_LOG(module, llc::FATAL)

#define CHECK(module, condition)                      \
  LLCOMPILER_CHECK_LOG(module, condition, llc::ERROR) \
      << __FILE__ << __LINE__ << #condition <<
#define CHECK_EQ(module, val1, val2) CHECK(module, val1 == val2)
#define CHECK_NE(module, val1, val2) CHECK(module, val1 != val2)
#define CHECK_LT(module, val1, val2) CHECK(module, val1 < val2)
#define CHECK_LE(module, val1, val2) CHECK(module, val1 <= val2)
#define CHECK_GT(module, val1, val2) CHECK(module, val1 > val2)
#define CHECK_GE(module, val1, val2) CHECK(module, val1 >= val2)

#define DCHECK(module, condition) \
  LLCOMPILER_CHECK_LOG(module, condition, llc::DEBUG)
#define DCHECK_EQ(module, val1, val2) DCHECK(module, val1 == val2)
#define DCHECK_NE(module, val1, val2) DCHECK(module, val1 != val2)
#define DCHECK_LT(module, val1, val2) DCHECK(module, val1 < val2)
#define DCHECK_LE(module, val1, val2) DCHECK(module, val1 <= val2)
#define DCHECK_GT(module, val1, val2) DCHECK(module, val1 > val2)
#define DCHECK_GE(module, val1, val2) DCHECK(module, val1 >= val2)

#define LOG(module, condition, lever) \
  LLCOMPILER_CHECK_LOG(module, condition, lever)
#define LOG_EQ(module, val1, val2, lever) \
  LLCOMPILER_CHECK_LOG(module, val1 == val2, lever)
#define LOG_NE(module, val1, val2, lever) \
  LLCOMPILER_CHECK_LOG(module, val1 != val2, lever)
#define LOG_LT(module, val1, val2, lever) \
  LLCOMPILER_CHECK_LOG(module, val1 < val2, lever)
#define LOG_LE(module, val1, val2, lever) \
  LLCOMPILER_CHECK_LOG(module, val1 <= val2, lever)
#define LOG_GT(module, val1, val2, lever) \
  LLCOMPILER_CHECK_LOG(module, val1 > val2, lever)
#define LOG_GE(module, val1, val2, lever) \
  LLCOMPILER_CHECK_LOG(module, val1 >= val2, lever)

#define TYPE(val) typeid(val).name()

#endif  // INCLUDE_LLCOMPILER_UTILS_LOGGER_H_
