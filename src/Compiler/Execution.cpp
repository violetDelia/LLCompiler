
#include "llcompiler/Compiler/Execution.h"

#include "llcompiler/Support/Logger.h"
#include "llvm/Support/DynamicLibrary.h"

namespace llc::compiler {

Execution::Execution() {}
void Execution::load(std::string shared_lib_path) {
  CHECK(llc::Executor, !is_initialized_) << "not supported currently";
  shared_lib_handle_ =
      llvm::sys::DynamicLibrary::getLibrary(shared_lib_path.c_str());
  CHECK(llc::Executor, shared_lib_handle_.isValid());
  entry_point_func_ = reinterpret_cast<entryPointFuncType>(
      shared_lib_handle_.getAddressOfSymbol("main"));
  CHECK(llc::Executor, entry_point_func_);
  is_initialized_ = true;
}

void Execution::run(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outs) {
  CHECK(llc::Executor, entry_point_func_);
  std::vector<void *> params;
  params.push_back(static_cast<void *>(nullptr));
  for (auto tensor : inputs) {
    params.push_back(static_cast<void *>(tensor->base));
    params.push_back(static_cast<void *>(tensor->data));
    params.push_back(static_cast<void *>(&tensor->offset));
    params.push_back(static_cast<void *>(tensor->size.data()));
    params.push_back(static_cast<void *>(tensor->stride.data()));
  }
  for (auto tensor : outs) {
    params.push_back(static_cast<void *>(tensor->base));
    params.push_back(static_cast<void *>(tensor->data));
    params.push_back(static_cast<void *>(&tensor->offset));
    params.push_back(static_cast<void *>(tensor->size.data()));
    params.push_back(static_cast<void *>(tensor->stride.data()));
  }
  entry_point_func_(static_cast<void **>(params.data()));
  return;
}
void Execution::run_with_symbols(std::vector<int64_t> &symbols, std::vector<Tensor *> &inputs,
                   std::vector<Tensor *> &outs) {
  CHECK(llc::Executor, entry_point_func_);
  std::vector<void *> params;
  if (symbols.size() != 0)
    params.push_back(static_cast<void *>(symbols.data()));
  else
    params.push_back(static_cast<void *>(nullptr));
  for (auto tensor : inputs) {
    params.push_back(static_cast<void *>(tensor->base));
    params.push_back(static_cast<void *>(tensor->data));
    params.push_back(static_cast<void *>(&tensor->offset));
    params.push_back(static_cast<void *>(tensor->size.data()));
    params.push_back(static_cast<void *>(tensor->stride.data()));
  }
  for (auto tensor : outs) {
    params.push_back(static_cast<void *>(tensor->base));
    params.push_back(static_cast<void *>(tensor->data));
    params.push_back(static_cast<void *>(&tensor->offset));
    params.push_back(static_cast<void *>(tensor->size.data()));
    params.push_back(static_cast<void *>(tensor->stride.data()));
  }
  entry_point_func_(static_cast<void **>(params.data()));
  return;
}

Execution::~Execution() {
  if (shared_lib_handle_.isValid())
    llvm::sys::DynamicLibrary::closeLibrary(shared_lib_handle_);
}

}  // namespace llc::compiler