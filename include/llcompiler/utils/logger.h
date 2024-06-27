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
#ifndef LLCOMPILER_UTILS_LOGGER_H
#define LLCOMPILER_UTILS_LOGGER_H
#include "llcompiler/core.h"

namespace llc::logger {
enum LOG_LEVER {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
};

/**
 * @brief 注册日志
 * @param  module 模块名
 * @param  root_path 日志根目录
 * @author 时光丶人爱 (1733535832@qq.com)
 * @date 2024-06-27
 */
void register_logger(const char *module, const char *root_path,
                     const LOG_LEVER lever);

/**
 * @brief 日志流,用来辅助实现宏
 * @author 时光丶人爱 (1733535832@qq.com)
 * @date 2024-06-28
 */
class Logger_Stream {
public:
  LLC_CONSTEXPR Logger_Stream(const char *module, LOG_LEVER level);

  /**
   * @brief 输入日志信息
   * @param  message 日志信息
   * @author 时光丶人爱 (1733535832@qq.com)
   * @date 2024-06-28
   */
  void operator<<(const char *message);

protected:
  const char *_module;
  LOG_LEVER _level;
};

} // namespace llc::logger
#ifdef LLCOMPILER_HAS_LOG
#define LLCOMPILER_INIT_LOGGER(module, root, lever) register_logger(module, root, lever);

#define LOG(module, lever) llc::logger::Logger_Stream(module, lever)
#define TRACE(module) llc::logger::Logger_Stream(module, llc::logger::TRACE)
#define INFO(module) llc::logger::Logger_Stream(module, llc::logger::INFO)
#define WARN(module) llc::logger::Logger_Stream(module, llc::logger::WARN)
#define ERROR(module) llc::logger::Logger_Stream(module, llc::logger::ERROR)
#define FATAL(module) llc::logger::Logger_Stream(module, llc::logger::FATAL)

#define CHECK(module, condition)                                               \
  if (condition)                                                               \
  ERROR(module)
#define CHECK_EQ(module, val1, val2) CHECK(module, val1 == val2)
#define CHECK_NE(module, val1, val2) CHECK(module, val1 != val2)
#define CHECK_LT(module, val1, val2) CHECK(module, val1 < val2)
#define CHECK_LE(module, val1, val2) CHECK(module, val1 <= val2)
#define CHECK_GT(module, val1, val2) CHECK(module, val1 > val2)
#define CHECK_GE(module, val1, val2) CHECK(module, val1 >= val2)

#define CHECK_LOG(module, condition, lever)                                    \
  if (condition)                                                               \
  LOG(module, lever)
#define LOG_EQ(module, val1, val2, lever) CHECK(module, val1 == val2, lever)
#define LOG_NE(module, val1, val2, lever) CHECK(module, val1 != val2, lever)
#define LOG_LT(module, val1, val2, lever) CHECK(module, val1 < val2, lever)
#define LOG_LE(module, val1, val2, lever) CHECK(module, val1 <= val2, lever)
#define LOG_GT(module, val1, val2, lever) CHECK(module, val1 > val2, lever)
#define LOG_GE(module, val1, val2, lever) CHECK(module, val1 >= val2, lever)
#else
#define LLCOMPILER_INIT_LOGGER(module, root, lever)

#define CHECK(module, condition)
#define CHECK_EQ(module, val1, val2)
#define CHECK_NE(module, val1, val2)
#define CHECK_LT(module, val1, val2)
#define CHECK_LE(module, val1, val2)
#define CHECK_GT(module, val1, val2)
#define CHECK_GE(module, val1, val2)

#define CHECK_LOG(module, condition, lever)
#define LOG_EQ(module, val1, val2, lever)
#define LOG_NE(module, val1, val2, lever)
#define LOG_LT(module, val1, val2, lever)
#define LOG_LE(module, val1, val2, lever)
#define LOG_GT(module, val1, val2, lever)
#define LOG_GE(module, val1, val2, lever)
#endif // LLCOMPILER_HAS_LOG
#endif // LLCOMPILER_UTILS_LOGGER_H
