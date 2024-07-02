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
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(sink_c);
  if (strcmp(root_path, "")) {
    auto now = std::chrono::system_clock::now();
    auto time_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&time_now), "%Y_%m_%/d%H:%M");
    auto time_str = time_ss.str().c_str();
    auto log_file = fmt::format("{}-{}/{}.log", root_path, time_str, module);
    sinks.push_back(std::make_shared<file_sink>(log_file, true));
  }
  auto log =
      std::make_shared<spdlog::logger>(module, sinks.begin(), sinks.end());
  log->set_level(static_cast<spdlog::level>(lever));
  spdlog::register_logger(log);
  INFO(GLOBAL) << "LOG_LEVER: " << static_cast<int>(lever);
  INFO(GLOBAL) << "LOG_ROOT_DIR: " << root_path;
  LOG(GLOBAL, strcmp(root_path, ""), LOG_LEVER::INFO) << "test" << "test2";
  LOG(GLOBAL, !strcmp(root_path, ""), LOG_LEVER::INFO) << "test" << "test2";
  CHECK(GLOBAL, strcmp(root_path, ""), LOG_LEVER::INFO) << "ctest" << "ctest2";
  // CHECK_NE(GLOBAL, strcmp(root_path, ""), LOG_LEVER::INFO) << "test";
}

LLC_CONSTEXPR Logger::Logger(const char *module, LOG_LEVER level)
    : module_(module), level_(level) {}

LLC_CONSTEXPR Logger::~Logger() {}

LLC_CONSTEXPR LoggerStream Logger::stream(const bool not_emit_message) {
  return LoggerStream(this, not_emit_message);
}

LLC_CONSTEXPR void Logger::info(const char *message) {
  std::shared_ptr<spdlog::logger> spd_logger = spdlog::get(this->module_);
  spd_logger->log(static_cast<spdlog::level>(this->level_), message);
}

template <class Ty>
LLC_CONSTEXPR NullStream &NullStream::operator<<(Ty val) {
  return *this;
}

LLC_CONSTEXPR LoggerStream::LoggerStream(Logger *log,
                                         const bool not_emit_message)
    : logger_(log), not_emit_message_(not_emit_message) {}

LLC_CONSTEXPR LoggerStream &LoggerStream::operator<<(const char *message) {
  if (!not_emit_message_) {
    message_ += message;
  }
  return *this;
}

LLC_CONSTEXPR LoggerStream &LoggerStream::operator<<(const std::string &str) {
  if (!not_emit_message_) {
    message_ += str;
  }
  return *this;
}

LLC_CONSTEXPR LoggerStream &LoggerStream::operator<<(const int value) {
  if (!not_emit_message_) {
    message_ += std::to_string(value);
  }
  return *this;
}

LLC_CONSTEXPR LoggerStream &LoggerStream::operator<<(const double value) {
  if (!not_emit_message_) {
    message_ += std::to_string(value);
  }
  return *this;
}

LLC_CONSTEXPR LoggerStream::~LoggerStream() {
  if (!not_emit_message_) return;
  logger_->info(message_.c_str());
}

}  // namespace llc::logger
