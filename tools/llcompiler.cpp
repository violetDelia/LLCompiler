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

#define LLCOMPILER_HAS_LOG
#include "llcompiler/llcompiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
using namespace llvm;
using namespace std;

static cl::OptionCategory cat("split-file Options");
static cl::opt<string> loggerRoot(cl::Positional, cl::desc("<input file>"),
                                  cl::init("-"));

cl::opt<string> OutputFilename("o", cl::desc("Specify output filename"),
                               cl::value_desc("filename"));
// cl::opt<string> InputFilename(cl::Positional, cl::desc("<input file>"),
// cl::init("-")); cl::opt<string> InputFilename(cl::Positional,
// cl::desc("<input file>"), cl::Required); cl::opt<string>
// InputFilename(cl::Positional, cl::Required, cl::desc("<input file>"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  LLCOMPILER_INIT_LOGGER(llc::OPTION, "", llc::logger::DEBUG);

  return 0;
}
