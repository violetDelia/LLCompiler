// #include "onnx/common/file_utils.h"
#define LLCOMPILER_HAS_LOG
#include "llcompiler/llcompiler.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

int main() {
  LLCOMPILER_INIT_LOGGER(llc::OPTION,
                         "E:/Tianyi_sync_folder/code/LLCompiler/log",
                         llc::logger::DEBUG);
  DEBUG(llc::OPTION) << "a" << "b";
  // log->info("Test log");
  //  auto sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("./log");
  //  spdlog::flush_on(spdlog::level::debug);
  //  llc::logger::register_logger("llcompiler", "", llc::logger::INFO);
  //  auto logger = spdlog::get("llcompiler");
  //  spdlog::flush_on(spdlog::level::debug);
  //  logger->info("LLCompiler: ONNX to LLCompiler");
}