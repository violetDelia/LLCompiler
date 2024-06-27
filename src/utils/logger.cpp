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
#include "llcompiler/utils/logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <memory>
#include <string>

namespace llc::logger {
void register_logger(const char *module, const char *root_path,
                     const LOG_LEVER lever) {
  spdlog::sinks_init_list sinks;
  if (root_path == "") {
    auto sink_c = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    spdlog::sinks_init_list sinks = {sink_c};
  } else {
    auto sink_c = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto sink_f = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        fmt::format("{}/{}.log", root_path, module));
    spdlog::sinks_init_list sinks = {sink_c, sink_f};
  }
  auto log =
      std::make_shared<spdlog::logger>(module, sinks.begin(), sinks.end());
  log->set_level(static_cast<spdlog::level>(lever));
  spdlog::register_logger(log);
}

LLC_CONSTEXPR
llc::logger::Logger_Stream::Logger_Stream(const char *module,
                                          llc::logger::LOG_LEVER level)
    : _module(module), _level(level) {};

void llc::logger::Logger_Stream::operator<<(const char *message) {
  auto logger = spdlog::get(this->_module);
  switch (this->_level) {
  case TRACE:
    logger->trace(message);
    break;
  case DEBUG:
    logger->debug(message);
    break;
  case INFO:
    logger->info(message);
    break;
  case WARN:
    logger->warn(message);
    break;
  case ERROR:
    logger->error(message);
    break;
  case FATAL:
    logger->critical(message);
    break;
  }
};
} // namespace llc::logger