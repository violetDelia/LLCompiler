
#include "llcompiler/Compiler/Command.h"

#include "llcompiler/Compiler/ToolPath.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
namespace llc::compiler {

std::string getToolPath(const std::string &tool) {
  return toolPathMap.at(tool);
}

Command::Command(std::string exe_path)
    : _path(std::move(exe_path)),
      _args({llvm::sys::path::filename(_path).str()}) {}

Command &Command::appendStr(const std::string &arg) {
  if (arg.size() > 0) _args.emplace_back(arg);
  return *this;
}

Command &Command::appendStrOpt(const std::optional<std::string> &arg) {
  if (arg.has_value()) _args.emplace_back(arg.value());
  return *this;
}

Command &Command::appendList(const std::vector<std::string> &args) {
  _args.insert(_args.end(), args.begin(), args.end());
  return *this;
}

Command &Command::resetArgs() {
  auto exe_file_name = _args.front();
  _args.clear();
  _args.emplace_back(exe_file_name);
  return *this;
}

void Command::exec(std::string work_dir) const {
  auto args = std::vector<llvm::StringRef>(_args.begin(), _args.end());

  llvm::SmallString<8> cur_work_dir;
  llvm::SmallString<8> new_work_dir(work_dir);
  llvm::sys::fs::current_path(cur_work_dir);
  llvm::sys::fs::make_absolute(cur_work_dir, new_work_dir);

  std::error_code ec = llvm::sys::fs::set_current_path(new_work_dir);
  CHECK(llc::GLOBAL, ec.value() == 0)
      << llvm::StringRef(new_work_dir).str() << ": " << ec.message()
      << ec.value() << "\n";
  INFO(llc::GLOBAL) << "[" << llvm::StringRef(new_work_dir).str() << "] "
                    << _path << ": " << llvm::join(args, " ") << "\n";

  std::string errMsg;
  CHECK(llc::GLOBAL, llvm::sys::ExecuteAndWait(
                         _path, llvm::ArrayRef(args),
                         /*Env=*/std::nullopt, /*Redirects=*/std::nullopt,
                         /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg) == 0)
      << llvm::join(args, " ") << "\n"
      << "Error message: " << errMsg << "\n"
      << "Program path: " << _path << "\n"
      << "Command execution failed."
      << "\n";

  llvm::sys::fs::set_current_path(cur_work_dir);
}
}  // namespace llc::compiler