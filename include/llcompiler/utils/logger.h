/*
 * @Author: lfr 1733535832@qq.com
 * @Date: 2024-06-27 00:14:56
 * @LastEditors: lfr 1733535832@qq.com
 * @LastEditTime: 2024-06-27 01:51:22
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

void register_logger(const char *name, const char *filename) {
  spdlog::sinks_init_list sinks;
  if (filename == "") {
    auto sink_c = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
    spdlog::sinks_init_list sinks = {sink_c};
  } else {
    auto sink_c = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
    auto sink_f = std::make_shared<spdlog::sinks::basic_file_sink_st>(
        fmt::format("{}/{}.log", filename, name));
    spdlog::sinks_init_list sinks = {sink_c, sink_f};
  }
  auto log = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
  spdlog::register_logger(log);
}

} // namespace llc::logger
