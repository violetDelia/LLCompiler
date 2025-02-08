# 前言

为了便于将整体编译工作进行模块化区分，方便调试和开发，最终将由LLVM源码实现的编译代码修改为通过使用编译工具实现编译链路。为此，在代码中配置了相应的工具链路径。目前使用的工具链依赖包括五个主要组件：llc-opt、llc-translate、opt（LLVM）、llc以及C++编译器。具体配置如下：
```c++
namespace llc::compiler {

namespace {
static const std::string OptPath = "@CMAKE_INSTALL_PREFIX@/bin/opt";
static const std::string LlcPath = "@CMAKE_INSTALL_PREFIX@/bin/llc";
static const std::string LlcOptPath = "@CMAKE_INSTALL_PREFIX@/bin/llc-opt";
static const std::string LlcTranslatePath =
    "@CMAKE_INSTALL_PREFIX@/bin/llc-translate";
static const std::string CXXPath = "@CMAKE_CXX_COMPILER@";

}  // namespace

static const std::map<std::string, std::string> toolPathMap = {
    {"opt", OptPath},
    {"llc", LlcPath},
    {"llc-opt", LlcOptPath},
    {"llc-translate", LlcTranslatePath},
    {"cxx", CXXPath}};
}
```

采用了直接在代码中调用命令行工具来完成编译流程的方法。

具体编译工具如下：

1. **llc-opt** ：作为首个工具，负责对MLIR IR进行转换与优化处理。
2. **llc-translate** ：作为第二个工具，用于将已降至LLVM Dialect的MLIR IR转换为LLVM IR。
3. **opt** ：作为第三个工具，对LLVM IR进行进一步的转换与优化，并将其编译为Bitcode格式。
4. **llc** ：作为第四个工具，将Bitcode文件编译为目标平台的目标文件（obj文件）。
5. **C++编译器** ：作为最后一步，将目标文件与相关库进行链接，最终生成一个动态链接库文件（so文件）。

## llc-opt

使用 `llc-opt`工具，将原始的MLIR IR经过100多个降级与优化Pass的处理，最终被转换为LLVM IR。

```c++
void LLCCompiler::optimizeMLIR(mlir::OwningOpRef<mlir::ModuleOp>& module,
                               CompileOptions options,
                               std::string opted_module_file) {
  preprocess_mlir_module(&module, options);
  if (options.pipeline == "transform") {
    runTransformPipeline(options, module);
  } else {
    UNIMPLEMENTED(llc::GLOBAL);
  }
  file::mlir_to_file(&module, opted_module_file.c_str());
};

void runTransformPipeline(CompileOptions& options,
                          mlir::OwningOpRef<mlir::ModuleOp>& module) {
  pipeline::TransformPipelineOptions pipleline_options;
  generatePipelineOptions(options, pipleline_options);
  // ********* process in mlir *********//
  mlir::PassManager pm(module.get()->getName());
  setIRDumpConfig(options, pm);
  pipeline::buildTransformPipeline(pm, pipleline_options);
  CHECK(MLIR, mlir::succeeded(pm.run(*module))) << "Failed to run pipeline";
}
```

## llc-translate

接下来，通过执行以下命令：

`llc-translate llc_module.opted.mlir -mlir-to-llvmir -o llc_module.ll `
可以将上述经过优化的MLIR IR转换为LLVM IR。

```c++
void LLCCompiler::translateMLIRToLLVMIR(std::string mlir_file,
                                        CompileOptions options,
                                        std::string llvm_ir_file) {
  Command translate_mlir(getToolPath("llc-translate"));
  translate_mlir.appendStr(mlir_file)
      .appendStr("-mlir-to-llvmir")
      .appendStr("-allow-unregistered-dialect")
      .appendList({"-o", llvm_ir_file})
      .exec();
}
```

## opt

opt 是对LLVM IR 进行优化的工具，使用 `opt llc_module.ll -o llc_module.opted.ll -O3 -S --march=x86-64 --mcpu=tigerlake --mtriple=x86_64-linux-gnu` 获得优化之后的llvm ir。

之后使用命令：`opt llc_module.opted.ll -o llc_module.bc --march=x86-64 --mcpu=tigerlake --mtriple=x86_64-linux-gnu `将其翻译为bitcode。

```c++
void LLCCompiler::optimizeLLVMIR(std::string llvm_ir_file,
                                 CompileOptions options,
                                 std::string opted_llvm_ir_file) {
  Command optimize_bitcode(getToolPath("opt"));
  optimize_bitcode.appendStr(llvm_ir_file)
      .appendList({"-o", opted_llvm_ir_file})
      .appendStr(getOptimizationLevelOption(options))
      .appendStr("-S")
      .appendStr(getTargetArchOption(options))
      .appendStr(getCPUOption(options))
      .appendStr(getMtripleOption(options));
  if (options.display_llvm_passes) {
    llvm::SmallString<128> all_passes_file(opted_llvm_ir_file);
    llvm::SmallString<128> llvm_ir_dir =
        llvm::sys::path::parent_path(opted_llvm_ir_file);
    llvm::sys::path::append(llvm_ir_dir, "llvm_dump");
    optimize_bitcode.appendStr("-print-before-all");
    optimize_bitcode.appendList({"-ir-dump-directory", llvm_ir_dir.c_str()});
  }
  optimize_bitcode.exec();
}

void LLCCompiler::translateLLVMIRToBitcode(std::string opted_llvm_ir_file,
                                           CompileOptions options,
                                           std::string llvm_bitcode_file) {
  Command translate_to_bitcode(getToolPath("opt"));
  translate_to_bitcode.appendStr(opted_llvm_ir_file)
      .appendList({"-o", llvm_bitcode_file})
      .appendStr(getTargetArchOption(options))
      .appendStr(getCPUOption(options))
      .appendStr(getMtripleOption(options))
      .exec();
}
```

## llc

得到bitcode 文件后，使用命令 `llc llc_module.bc -o llc_module.o -filetype=obj -relocation-model=pic --march=x86-64 --mcpu=tigerlake --mtriple=x86_64-linux-gnu`将bitcode文件编译为obj文件。

```c++
void LLCCompiler::translateBitcodeToObject(std::string llvm_bitcode_file,
                                           CompileOptions options,
                                           std::string object_file) {
  Command bitcode_to_object(getToolPath("llc"));
  bitcode_to_object.appendStr(llvm_bitcode_file)
      .appendList({"-o", object_file})
      .appendStr("-filetype=obj")
      .appendStr("-relocation-model=pic")
      .appendStr(getTargetArchOption(options))
      .appendStr(getCPUOption(options))
      .appendStr(getMtripleOption(options))
      .exec();
}
```

最后将obj文件链接相关的库文件，将模型编译为一个so文件，
`g++ llc_module.o -o llc_module.so -shared -fPIC -lmlir_c_runner_utils -L/lib -L/llvm/lib`

```c++
void LLCCompiler::generateSharedLib(std::vector<std::string> objs,
                                    CompileOptions options,
                                    std::string shared_lib_file) {
  Command link(getToolPath("cxx"));
  link.appendList(objs)
      .appendList({"-o", shared_lib_file})
      .appendList({"-shared", "-fPIC"});
  for (auto lib : options.libs) {
    link.appendStr("-l" + lib);
  }
  for (auto lib_dir : options.libdirs) {
    link.appendStr("-L" + lib_dir);
  }
  link.exec();
}
```

## 运行
将模型编译为一个so 文件后，加载该so文件，找到入口函数的地址，就可以运行模型了。相较于之前直接使用llvm.orc的Jit引擎，要简单且高效很多。
```c++
using entryPointFuncType = void (*)(void**);

class Execution {
 public:
  explicit Execution();
  ~Execution();

  void run(std::vector<Tensor*>& inputs, std::vector<Tensor*>& outs);

  void run_with_symbols(std::vector<int64_t>& symbols, std::vector<Tensor*>& inputs,
           std::vector<Tensor*>& outs);

  void load(std::string shared_lib_path);

 protected:
  bool is_initialized_ = false;
  llvm::sys::DynamicLibrary shared_lib_handle_;
  entryPointFuncType entry_point_func_;
};

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
```
