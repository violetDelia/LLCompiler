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

#include "llcompiler/Support/Logger.h"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Enums.h"
#include "spdlog/common.h"
#include "spdlog/logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"

namespace llc {

/**********  log module define  **********/
const char *GLOBAL = "global";
const char *IMPORTER = "importer";
const char *UTILITY = "utility";
const char *MLIR = "mlir";
const char *MLIR_PASS = "mlir_pass";
const char *DEBUG = "debug";
const char *SymbolInfer = "symbol_infer";
const char *Entrance_Module = "entrance_module";
const char *Executor = "executor";
}  // namespace llc

namespace llc::logger {

void register_logger(const char *module, const LoggerOption &option) {
  using console_sink = spdlog::sinks::stdout_sink_st;
  using file_sink = spdlog::sinks::basic_file_sink_st;
  auto sink_c = std::make_shared<console_sink>();
  sink_c->set_level(static_cast<spdlog::level>(llc::LogLevel::warn));
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(sink_c);
  std::string log_file;
  auto path = option.path.c_str();
  const bool save_log = strcmp(path, "");
  if (save_log) {
    auto now = std::chrono::system_clock::now();
    auto time_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&time_now), "%Y_%m_%d_%H_%M");
    // auto log_dir = fmt::format("{}/log_{}", path, time_ss.str().c_str());
    auto log_dir = fmt::format("{}", path);
    if (!std::filesystem::exists(log_dir)) {
      std::filesystem::create_directories(log_dir);
    }
    log_file = fmt::format("{}/{}.log", log_dir, module);
    auto sink_f = std::make_shared<file_sink>(log_file.c_str(), true);
    sinks.push_back(sink_f);
  }
  auto logger = spdlog::get(module);
  if (logger) {
    logger->set_level(static_cast<spdlog::level>(option.level));
    logger->sinks().clear();
    logger->sinks().insert(logger->sinks().end(), sinks.begin(), sinks.end());
    INFO(GLOBAL) << "replace log module: " << module
                 << "(lever:" << llc::stringifyLogLevel(option.level).str()
                 << ")"
                 << " -> " << log_file;
  } else {
    auto log =
        std::make_shared<spdlog::logger>(module, sinks.begin(), sinks.end());
    log->set_pattern("[%T] [%n] [%^%l%$] %v");
    log->set_level(static_cast<spdlog::level>(option.level));
    spdlog::register_logger(log);
    INFO(GLOBAL) << "regist log module: " << module
                 << "(lever:" << llc::stringifyLogLevel(option.level).str()
                 << ")"
                 << " -> " << log_file;
  }
}

Logger::Logger(const char *module, const llc::LogLevel level)
    : module_(module), level_(level) {}

Logger::~Logger() {}

LoggerStream Logger::stream(const bool emit_message) {
  return LoggerStream(this, emit_message);
}

void Logger::info(const char *message) {
  std::shared_ptr<spdlog::logger> logger = spdlog::get(this->module_);
  if (logger) {
    logger->log(static_cast<spdlog::level>(this->level_), message);
  } else {
    if (this->level_ >= llc::LogLevel::error)
      spdlog::log(static_cast<spdlog::level>(this->level_), message);
  }
}

LoggerStream::LoggerStream(Logger *log, const bool emit_message)
    : logger_(log), emit_message_(emit_message) {}

LoggerStream::~LoggerStream() {
  if (emit_message_) {
    logger_->info(message_.str().c_str());
  }
  return;
}

}  // namespace llc::logger
