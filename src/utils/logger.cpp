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
  std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> sink_c =
      std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  std::shared_ptr<spdlog::sinks::basic_file_sink_mt> sink_f;
  if (root_path != "") {
    auto module_file = fmt::format("{}/{}.log", root_path, module);
    sink_f =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(module_file, true);
  }
  spdlog::sinks_init_list sinks = {sink_c, sink_f};
  auto log =
      std::make_shared<spdlog::logger>(module, sinks.begin(), sinks.end());
  log->set_level(static_cast<spdlog::level>(lever));
  spdlog::register_logger(log);
}

LLC_CONSTEXPR LoggerStream::LoggerStream(Logger *log) : logger_(log) {};
LLC_CONSTEXPR LoggerStream &LoggerStream::operator<<(const char *message) {
  message_ << message;
  return *this;
}

LLC_CONSTEXPR LoggerStream::~LoggerStream() {
  logger_->info(message_.str().c_str());
}

LLC_CONSTEXPR Logger::Logger(const char *module, LOG_LEVER level)
    : module_(module), level_(level) {}

LLC_CONSTEXPR LoggerStream Logger::stream() { return LoggerStream(this); }

LLC_CONSTEXPR void Logger::info(const char *message) {
  std::shared_ptr<spdlog::logger> spd_logger = spdlog::get(this->module_);
  spd_logger->log(static_cast<spdlog::level>(this->level_), message);
}

LLC_CONSTEXPR Logger::~Logger(){};

LLC_CONSTEXPR NullStream &NullStream::operator<<(const char *message) {
  return *this;
}

} // namespace llc::logger