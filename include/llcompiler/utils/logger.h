/*
 * @Author: lfr 1733535832@qq.com
 * @Date: 2024-06-27 00:14:56
 * @LastEditors: lfr 1733535832@qq.com
 * @LastEditTime: 2024-06-27 02:07:51
 * @FilePath: \LLCompiler\include\llcompiler\utils\logger.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
 * @brief 日志
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024  时光丶人爱
 *
 */
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <memory>

namespace llc::logger {
enum LOG_LEVER {
  LOG_LEVER_TRACE = 0,
  LOG_LEVER_DEBUG = 1,
  LOG_LEVER_INFO = 2,
  LOG_LEVER_WARN = 3,
  LOG_LEVER_ERROR = 4,
  LOG_LEVER_FATAL = 5,
};

void register_logger(const char *module, const char *filename) {
  spdlog::sinks_init_list sinks;
  if (filename == "") {
    auto sink_c = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
    spdlog::sinks_init_list sinks = {sink_c};
  } else {
    auto sink_c = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
    auto sink_f = std::make_shared<spdlog::sinks::basic_file_sink_st>(
        fmt::format("{}/{}.log", filename, module));
    spdlog::sinks_init_list sinks = {sink_c, sink_f};
  }
  auto log = std::make_shared<spdlog::logger>(module, sinks.begin(), sinks.end());
  spdlog::register_logger(log);
}
} // namespace llc::logger
// #define LOG_ERROR(msg...)                                           \
//     if (LITE_ERROR >= g_log_level) {                                \
//         __tinynn_log__("TinyNN ERROR:%s@%d: ", __func__, __LINE__); \
//         __tinynn_log__(msg);                                        \
//     }
// #define LOG_ERROR_NO_PREFIX(msg...)  \
//     if (LITE_ERROR >= g_log_level) { \
//         __tinynn_log__(msg);         \
//     }
// #define LOG_WARNING(msg...)                                        \
//     if (LITE_WARN >= g_log_level) {                                \
//         __tinynn_log__("TinyNN WARN:%s@%d: ", __func__, __LINE__); \
//         __tinynn_log__(msg);                                       \
//     }
// #define LOG_INFO(msg...)                                           \
//     if (LITE_INFO >= g_log_level) {                                \
//         __tinynn_log__("TinyNN INFO:%s@%d: ", __func__, __LINE__); \
//         __tinynn_log__(msg);                                       \
//     }
// #define LOG_DEBUG(msg...)                                           \
//     if (LITE_DEBUG >= g_log_level) {                                \
//         __tinynn_log__("TinyNN DEBUG:%s@%d: ", __func__, __LINE__); \
//         __tinynn_log__(msg);                                        \
//     }
// #define LOG_DEBUG_NO_PREFIX(msg...)  \
//     if (LITE_DEBUG >= g_log_level) { \
//         __tinynn_log__(msg);         \
//     }
