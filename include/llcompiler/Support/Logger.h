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
 * @file Logger.h
 * @brief Logger and macros for logging
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024  时光丶人爱
 *
 */
#ifndef INCLUDE_LLCOMPILER_SUPPORT_LOGGER_H_
#define INCLUDE_LLCOMPILER_SUPPORT_LOGGER_H_
#include <cstddef>
#include <sstream>
#include <string>

#include "llcompiler/Support/Enums.h"
namespace llc {

/**********  log module extern  **********/
extern const char *GLOBAL;
extern const char *IMPORTER;
extern const char *UTILITY;
extern const char *MLIR;
extern const char *MLIR_PASS;
extern const char *DEBUG;
extern const char *SymbolInfer;
extern const char *Entrance_Module;
extern const char *Executor;
};  // namespace llc

namespace llc::logger {

struct LoggerOption {
  std::string path;
  llc::LogLevel level;
};

void register_logger(const char *module, const LoggerOption &option);

class LoggerStream;

class Logger {
 public:
  Logger(const char *module, const llc::LogLevel level);
  virtual ~Logger();
  LoggerStream stream(const bool emit_message);
  void info(const char *message);

 protected:
  const char *module_;
  const llc::LogLevel level_;
};

class NullStream {
 public:
  template <class Ty>
  NullStream &operator<<(const Ty val);
};

class LoggerStream {
 public:
  LoggerStream(Logger *log, const bool emit_message);
  template <class Ty>
  LoggerStream &operator<<(const Ty value);
  virtual ~LoggerStream();

 protected:
  std::stringstream message_;
  Logger *logger_;
  const bool emit_message_;
};

template <class Ty>
LoggerStream &LoggerStream::operator<<(const Ty message) {
  if (emit_message_) {
    message_ << message;
  }
  return *this;
}

template <class Ty>
NullStream &NullStream::operator<<(const Ty val) {
  return *this;
}
}  // namespace llc::logger

#ifdef LLCOMPILER_HAS_LOG
#define LLCOMPILER_INIT_LOGGER(module, root, lever) \
  ::llc::logger::register_logger(module, root, lever);
#define LLCOMPILER_LOG(module, lever) \
  ::llc::logger::Logger(module, lever).stream(true)
#define LLCOMPILER_CHECK_LOG(module, condition, lever)               \
  ::llc::logger::Logger(module, static_cast<::llc::LogLevel>(lever)) \
      .stream(!condition)
#else
#define LLCOMPILER_INIT_LOGGER(module, root, lever)
#define LLCOMPILER_LOG(module, lever) ::llc::logger::NullStream()
#define LLCOMPILER_CHECK_LOG(module, condition, lever) \
  ::llc::logger::NullStream()
#endif  // LLCOMPILER_HAS_LOG

#define DINFO LLCOMPILER_LOG(llc::DEBUG, ::llc::LogLevel::error)
#define DEBUG(module) LLCOMPILER_LOG(module, ::llc::LogLevel::debug)
#define INFO(module) LLCOMPILER_LOG(module, ::llc::LogLevel::info)
#define WARN(module)                            \
  LLCOMPILER_LOG(module, ::llc::LogLevel::warn) \
      << __FILE__ << "<" << __LINE__ << ">: \n\t"
#define WRONG(module)                            \
  LLCOMPILER_LOG(module, ::llc::LogLevel::error) \
      << __FILE__ << "<" << __LINE__ << ">: \n\t"
#define FATAL(module)                            \
  LLCOMPILER_LOG(module, ::llc::LogLevel::fatal) \
      << __FILE__ << "<" << __LINE__ << ">: \n\t"

#define print_info LLCOMPILER_LOG(::llc::GLOBAL, ::llc::LogLevel::error)

#define CHECK(module, condition)                                  \
  LLCOMPILER_CHECK_LOG(module, condition, ::llc::LogLevel::error) \
      << #condition << " : " << __FILE__ << "<" << __LINE__ << "> \n\t"
#define CHECK_EQ(module, val1, val2) CHECK(module, (val1 == val2))
#define CHECK_NE(module, val1, val2) CHECK(module, (val1 != val2))
#define CHECK_LT(module, val1, val2) CHECK(module, (val1 < val2))
#define CHECK_LE(module, val1, val2) CHECK(module, (val1 <= val2))
#define CHECK_GT(module, val1, val2) CHECK(module, (val1 > val2))
#define CHECK_GE(module, val1, val2) CHECK(module, (val1 >= val2))

#define DCHECK(module, condition) \
  LLCOMPILER_CHECK_LOG(module, condition, ::: llc::LogLevel::debug)
#define DCHECK_EQ(module, val1, val2) DCHECK(module, (val1 == val2))
#define DCHECK_NE(module, val1, val2) DCHECK(module, (val1 != val2))
#define DCHECK_LT(module, val1, val2) DCHECK(module, (val1 < val2))
#define DCHECK_LE(module, val1, val2) DCHECK(module, (val1 <= val2))
#define DCHECK_GT(module, val1, val2) DCHECK(module, (val1 > val2))
#define DCHECK_GE(module, val1, val2) DCHECK(module, (val1 >= val2))

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

#define UNIMPLEMENTED(module) \
  WRONG(module) << "function [" << __func__ << "] Unimplemented!"

#define WARN_UNIMPLEMENTED(module) \
  WARN(module) << "function [" << __func__ << "] Unimplemented!"

#define INFO_UNIMPLEMENTED(module)                         \
  INFO(module) << __FILE__ << "<" << __LINE__ << ">: \n\t" \
               << "function [" << __func__ << "] Unimplemented!"
#endif  // INCLUDE_LLCOMPILER_SUPPORT_LOGGER_H_
