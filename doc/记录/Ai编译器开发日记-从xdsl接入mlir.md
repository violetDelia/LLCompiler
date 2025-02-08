# 从xDSL接入MLR

我们通过遍历fx_graph 创建了xDSL的IR图，下一步就是将python中的xDSL的IR接入到C++的MLIR框架中，正式开始编译流程。

## pythonbind11

首先定义一个在C++上定义一个入口函数do_compiler：传入xDSL的字符串，以及一些编译的选项。
在这个函数内部里面，会将Module 翻译成mlir 格式的 Moduel，然后进行数个Pass转换为LLVM的IR，同时将设备端的IR翻译为硬件运行指令集，最后链接必要的运行时库和算子库，这样整个模型的编译流程就结束了。最后返回一个执行器，执行器的作用是运行已经编译好的模型文件，从python中接受tensor数据进行执行编译后的模型并返回结果。

```cpp
ExecuteEngine * (const str * xdsl_module, CompileOptions * option,...); 
```

然后通过pythondbind11将这个函数打包为动态库在python中调用。

打包的名称为llcompiler_;

```C++
#include "Compiler/Entrance.h"

#include <iostream>

#include "pybind11/pybind11.h"
namespace llc::compiler {

PYBIND11_MODULE(llcompiler_, llcompiler_) {
  auto entrance = llcompiler_.def_submodule("entrance");
  entrance.doc() = "entrance for compiler";  // optional module docstring
  entrance.def("do_compile", &do_compile, "");
}
}  // namespace llc::compiler

```

最后在steup.py文件里面定义C++扩展库

```python
ext_modules = []
source_files = glob.glob("{}/*.cpp".format(PYBIND_DIR), recursive=True)
libraries = ["LLCompiler"]
ext_modules = [
Pybind11Extension(
"llcompiler_",  # depends on the structure of your package
source_files,
# Example: passing in the version to the compiled code
include_dirs=INCLUDE_DIRS,
library_dirs=LIBRARY_DIRS,
runtime_library_dirs=RUNTIME_LIBRARY_DIRS,
libraries=libraries,
language="C++",
define_macros=[("VERSION_INFO", VERSION)],
),
]
```

这样，就可以在python中调用C++中定义的函数do_compiler了,下图是自定义编译器的实际执行函数，self.importer会将解析传入模型的模型生成xDSL的Module，之后将Module通过字符串的形式传入绑定的C++执行函数。

```
from llcompiler_.entrance import do_compile

def compiler(self, model: Any):
self._mlir_module = self.importer(model)
if self.vebose_first_ir:
print(self._mlir_module)
do_compile(
self._mlir_module.__str__(),
self.mode,
self.target,
self.ir_tree_dir,
self.log_path,
self.log_level
)
return model
```

## 将字符串解析为MLIR的Module

以下是模型的原始定义：

```python
class Base(nn.Module):
def __init__(self):
super(Base, self).__init__()
self.conv_layer1 = nn.Conv2d(
3, 10, stride=2, kernel_size=5, padding=2, dilation=5
)
self.conv_layer2 = nn.Conv2d(10, 3, kernel_size=5, padding=5, bias=True)
self.batch = nn.BatchNorm2d(100)
self.cf = nn.Linear(int((224 - 17) / 2 + 7), 2)

def forward(self, x: torch.Tensor):
x = x.reshape(x.shape[3],x.shape[2],x.shape[0],x.shape[1])
x = x.reshape(x.shape[2],x.shape[3],x.shape[1],x.shape[0])
x = self.conv_layer1(x)
x1 = x + x
c = 2 + 2 * 5 / 3
x = x / c
x2 = x + x1 + x * x
x = self.conv_layer2(x2 + x1)
x = self.cf(x + x * x + x / 2)
return x
```

将其转为xDSL后的Module后如下所示：

```mlir
builtin.module attributes  {"builtin.gloabal_layout" = "NCHW"} {
  func.func @main(%0 : tensor<?x3x?x?xf32>) -> tensor<?x3x?x2xf32>  attributes {"entrance"}{
%1 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T14:45:30.821309+08:00/L__self___conv_layer1.weight.npy"} : () -> tensor<10x3x5x5xf32>
%2 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T14:45:30.821309+08:00/L__self___conv_layer1.bias.npy"} : () -> tensor<10xf32>
%3 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T14:45:30.821309+08:00/L__self___conv_layer2.weight.npy"} : () -> tensor<3x10x5x5xf32>
%4 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T14:45:30.821309+08:00/L__self___conv_layer2.bias.npy"} : () -> tensor<3xf32>
%5 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T14:45:30.821309+08:00/L__self___cf.weight.npy"} : () -> tensor<2x110xf32>
%6 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T14:45:30.821309+08:00/L__self___cf.bias.npy"} : () -> tensor<2xf32>
%7 = "llh.torch_symbolic_int"() {"sym_name" = "s0"} : () -> i64
%8 = "llh.torch_symbolic_int"() {"sym_name" = "s2"} : () -> i64
"llh.symbolic_bind"(%0, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%9 = "llh.constant"() {"value" = 3 : i64} : () -> i64
%10 = "llh.dim"(%0, %9) : (tensor<?x3x?x?xf32>, i64) -> i64
%11 = "llh.constant"() {"value" = 2 : i64} : () -> i64
%12 = "llh.dim"(%0, %11) : (tensor<?x3x?x?xf32>, i64) -> i64
%13 = "llh.constant"() {"value" = 0 : i64} : () -> i64
%14 = "llh.dim"(%0, %13) : (tensor<?x3x?x?xf32>, i64) -> i64
%15 = "llh.constant"() {"value" = 1 : i64} : () -> i64
%16 = "llh.dim"(%0, %15) : (tensor<?x3x?x?xf32>, i64) -> i64
%17 = "llh.reshape"(%0, %10, %12, %14, %16) : (tensor<?x3x?x?xf32>, i64, i64, i64, i64) -> tensor<?x?x?x3xf32>
"llh.symbolic_bind"(%17, %8, %7) {"expressions" = affine_map<()[s0, s1] -> (s0, s0, s1, 3)>} : (tensor<?x?x?x3xf32>, i64, i64) -> ()
%18 = "llh.constant"() {"value" = 2 : i64} : () -> i64
%19 = "llh.dim"(%17, %18) : (tensor<?x?x?x3xf32>, i64) -> i64
%20 = "llh.constant"() {"value" = 3 : i64} : () -> i64
%21 = "llh.dim"(%17, %20) : (tensor<?x?x?x3xf32>, i64) -> i64
%22 = "llh.constant"() {"value" = 1 : i64} : () -> i64
%23 = "llh.dim"(%17, %22) : (tensor<?x?x?x3xf32>, i64) -> i64
%24 = "llh.constant"() {"value" = 0 : i64} : () -> i64
%25 = "llh.dim"(%17, %24) : (tensor<?x?x?x3xf32>, i64) -> i64
%26 = "llh.reshape"(%17, %19, %21, %23, %25) : (tensor<?x?x?x3xf32>, i64, i64, i64, i64) -> tensor<?x3x?x?xf32>
"llh.symbolic_bind"(%26, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, s1, ((s1 floordiv s0) * s0))>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%27 = "llh.conv_bias"(%26, %1, %2) {"dilation" = array<i64: 5, 5>, "pad" = array<i64: 2, 2, 2, 2>, "group" = 1 : i64, "kernel_shape" = array<i64: 5, 5>, "stride" = array<i64: 2, 2>} : (tensor<?x3x?x?xf32>, tensor<10x3x5x5xf32>, tensor<10xf32>) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%27, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%28 = "llh.add"(%27, %27) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%28, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%29 = "llh.constant"() {"value" = 5.333333e+00 : f32} : () -> f32
%30 = "llh.div"(%27, %29) : (tensor<?x10x?x?xf32>, f32) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%30, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%31 = "llh.add"(%30, %28) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%31, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%32 = "llh.mul"(%30, %30) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%32, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%33 = "llh.add"(%31, %32) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%33, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%34 = "llh.add"(%33, %28) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
"llh.symbolic_bind"(%34, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 10, (((s1 + -17) floordiv 2) + 1), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 1))>} : (tensor<?x10x?x?xf32>, i64, i64) -> ()
%35 = "llh.conv_bias"(%34, %3, %4) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 5, 5, 5, 5>, "group" = 1 : i64, "kernel_shape" = array<i64: 5, 5>, "stride" = array<i64: 1, 1>} : (tensor<?x10x?x?xf32>, tensor<3x10x5x5xf32>, tensor<3xf32>) -> tensor<?x3x?x?xf32>
"llh.symbolic_bind"(%35, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, (((s1 + -17) floordiv 2) + 7), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 7))>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%36 = "llh.mul"(%35, %35) : (tensor<?x3x?x?xf32>, tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
"llh.symbolic_bind"(%36, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, (((s1 + -17) floordiv 2) + 7), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 7))>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%37 = "llh.add"(%35, %36) : (tensor<?x3x?x?xf32>, tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
"llh.symbolic_bind"(%37, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, (((s1 + -17) floordiv 2) + 7), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 7))>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%38 = "llh.constant"() {"value" = 2 : i64} : () -> i64
%39 = "llh.div"(%35, %38) : (tensor<?x3x?x?xf32>, i64) -> tensor<?x3x?x?xf32>
"llh.symbolic_bind"(%39, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, (((s1 + -17) floordiv 2) + 7), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 7))>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%40 = "llh.add"(%37, %39) : (tensor<?x3x?x?xf32>, tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
"llh.symbolic_bind"(%40, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, (((s1 + -17) floordiv 2) + 7), (((((s1 floordiv s0) * s0) + -17) floordiv 2) + 7))>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
%41 = "llh.transpose"(%5) {"perms" = array<i64: 1, 0>} : (tensor<2x110xf32>) -> tensor<110x2xf32>
%42 = "llh.matmul"(%40, %41) : (tensor<?x3x?x?xf32>, tensor<110x2xf32>) -> tensor<?x3x?x2xf32>
%43 = "llh.add"(%42, %6) : (tensor<?x3x?x2xf32>, tensor<2xf32>) -> tensor<?x3x?x2xf32>
"llh.symbolic_bind"(%43, %7, %8) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, (((s1 + -17) floordiv 2) + 7), 2)>} : (tensor<?x3x?x2xf32>, i64, i64) -> ()
func.return %43 : tensor<?x3x?x2xf32>
  }
}
```

通过调用MLIR封装好的接口很方便的就可以将字符串解析为MLIR的IR图：

```cpp
void str_to_mlir_module(mlir::MLIRContext& context,
mlir::OwningOpRef[mlir::ModuleOp](mlir::ModuleOp)& module,
const char* str) {
  llvm::ErrorOr[std::unique_ptrllvm::memorybuffer<> fileOrErr =
  llvm::MemoryBuffer::getMemBuffer(str, "xdsl_module");
  if (std::error_code ec = fileOrErr.getError()) {
FATAL(UTILITY) << "load xdsl module fatal error!";
return;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile](std::unique_ptr%3Cllvm::MemoryBuffer)[mlir::ModuleOp](mlir::ModuleOp)(sourceMgr, {&context,false});
  if (!module) {
FATAL(UTILITY) << "parse xdsl module fatal error!";
return;
  }
  return;
}
```

至此，我们前端的工作就暂时告一段落，之后就可以在MLIR中对计算图进行编译和优化了。
