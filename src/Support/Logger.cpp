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

#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "spdlog/common.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace llc::logger {

void register_logger(const char *module, const char *root_path,
                     const LOG_LEVER lever) {
  using console_sink = spdlog::sinks::stdout_color_sink_mt;
  using file_sink = spdlog::sinks::basic_file_sink_st;
  auto sink_c = std::make_shared<console_sink>();
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(sink_c);
  auto now = std::chrono::system_clock::now();
  auto time_now = std::chrono::system_clock::to_time_t(now);
  std::stringstream time_ss;
  time_ss << std::put_time(std::localtime(&time_now), "%Y_%m_%d_%H_%M");
  std::string log_dir =
      fmt::format("{}/log_{}", root_path, time_ss.str().c_str());
  std::string log_file;
  const bool save_log = strcmp(root_path, "");
  if (save_log) {
    log_file = fmt::format("{}/{}.log", log_dir, module);
    sinks.push_back(std::make_shared<file_sink>(log_file.c_str(), true));
  }
  auto log =
      std::make_shared<spdlog::logger>(module, sinks.begin(), sinks.end());
  log->set_level(static_cast<spdlog::level::level_enum>(lever));
  spdlog::register_logger(log);
  if (std::filesystem::exists(log_dir)) {
    CHECK(GLOBAL, std::filesystem::create_directories(log_dir))
        << "create file " << log_dir.c_str() << " failed!";
  }
  INFO(GLOBAL) << "regist log module: " << module
               << "(lever:" << logger::log_lever_to_str(lever) << ")" << " -> "
               << log_file;
}

Logger::Logger(const char *module, const LOG_LEVER level)
    : module_(module), level_(level) {}

Logger::~Logger() {}

LoggerStream Logger::stream(const bool emit_message) {
  return LoggerStream(this, emit_message);
}

void Logger::info(const char *message) {
  std::shared_ptr<spdlog::logger> spd_logger = spdlog::get(this->module_);
  spd_logger->log(static_cast<spdlog::level::level_enum>(this->level_), message);
}

template <class Ty>
NullStream &NullStream::operator<<(Ty val) {
  return *this;
}

LoggerStream::LoggerStream(Logger *log, const bool emit_message)
    : logger_(log), emit_message_(emit_message) {}

LoggerStream &LoggerStream::operator<<(const char *message) {
  if (emit_message_) {
    message_ += message;
  }
  return *this;
}

LoggerStream &LoggerStream::operator<<(const std::string &str) {
  if (emit_message_) {
    message_ += str;
  }
  return *this;
}

LoggerStream &LoggerStream::operator<<(const int value) {
  if (emit_message_) {
    message_ += std::to_string(value);
  }
  return *this;
}

LoggerStream &LoggerStream::operator<<(const std::int64_t value) {
  if (emit_message_) {
    message_ += std::to_string(value);
  }
  return *this;
}

LoggerStream &LoggerStream::operator<<(const double value) {
  if (emit_message_) {
    message_ += std::to_string(value);
  }
  return *this;
}

LoggerStream::~LoggerStream() {
  if (emit_message_) {
    logger_->info(message_.c_str());
  }
  return;
}

}  // namespace llc::logger
