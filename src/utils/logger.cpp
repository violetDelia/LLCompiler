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
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "llcompiler/utils/logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace llc::logger {
void register_logger(const char *module, const char *root_path,
                     const LOG_LEVER lever) {
  using console_sink = spdlog::sinks::stdout_color_sink_mt;
  using file_sink = spdlog::sinks::basic_file_sink_mt;

  auto sink_c = std::make_shared<console_sink>();
  std::shared_ptr<file_sink> sink_f;
  if (strcmp(root_path, "")) {
    auto now = std::chrono::system_clock::now();
    auto time_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&time_now), "%Y_%m_%/d%H:%M");
    auto time_str = time_ss.str().c_str();
    auto log_file = fmt::format("{}-{}/{}.log", root_path, time_str, module);
    sink_f = std::make_shared<file_sink>(log_file, true);
  }
  spdlog::sinks_init_list sinks = {sink_c, sink_f};
  auto log = std::make_shared<spdlog::logger>(module, sinks);
  log->set_level(static_cast<spdlog::level>(lever));
  spdlog::register_logger(log);
  // INFO(LLC_OPTION) << "LOG_LEVER: ";
  // INFO(LLC_OPTION) << "LOG_ROOT_DIR: ";
}

LLC_CONSTEXPR LoggerStream::LoggerStream(Logger *log) : logger_(log) {
  std::cout << "LoggerStream" << std::endl;
}

LLC_CONSTEXPR LoggerStream &LoggerStream::operator<<(const char *message) {
  message_ << message;
  return *this;
}

LLC_CONSTEXPR LoggerStream::~LoggerStream() {
  logger_->info(message_.str().c_str());
  std::cout << "~LoggerStream" << std::endl;
}

LLC_CONSTEXPR Logger::Logger(const char *module, LOG_LEVER level)
    : module_(module), level_(level) {
  std::cout << "Logger" << std::endl;
}

LLC_CONSTEXPR LoggerStream Logger::stream() { return LoggerStream(this); }

LLC_CONSTEXPR void Logger::info(const char *message) {
  std::cout << "info" << std::endl;
  std::shared_ptr<spdlog::logger> spd_logger = spdlog::get(this->module_);
  std::cout << "message: " << message << std::endl;
  spd_logger->info(message);
  std::cout << "out info" << std::endl;
}

LLC_CONSTEXPR Logger::~Logger() { std::cout << "~Logger" << std::endl; }

LLC_CONSTEXPR NullStream &NullStream::operator<<(const char *message) {
  return *this;
}

}  // namespace llc::logger
