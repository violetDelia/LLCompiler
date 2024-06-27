// #include "onnx/common/file_utils.h"
#define LLCOMPILER_HAS_LOG
#include "llcompiler/llcompiler.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

int main() {
    auto sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("./log");
    spdlog::flush_on(spdlog::level::debug);

  // llc::logger::register_logger("llcompiler", "", llc::logger::INFO);
  //   auto logger = spdlog::get("llcompiler");
  //   logger->info("LLCompiler: ONNX to LLCompiler");

}