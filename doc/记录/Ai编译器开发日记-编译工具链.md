# 前言

为了能够方便将整体的编译工作区分开来，方便调试和维护，最终决定利用编译工具来实现编译的整个链路。
因此在代码中配置了相关工具链的路径，目前使用的工具依赖有5个,分别是：llc-opt,llc-translate,opt(llvm),llc,以及C++编译器。
具体配置如下：

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
在代码中直接调用命令行来实现编译的工作（借鉴了onnx-mlir的做法）。

第一个工具llc-opt是用来进行MLIR IR上的变换与优化。
第二个工具llc-translate用来将已经下降到llvm dialect 上的 MLIR IR 翻译为 LLVM IR。
第三个工具opt 对llvm ir 进行变换与优化，并将llvm ir 翻译为bitcode。。
第四个工具llc 将bitcode 文件翻译为 obj文件。
最后是Cxx的编译器，用来将obj文件打包起来，链接相关的库，将模型编译为一个so文件。

# llc-opt
```mlir
builtin.module {
  func.func @main(%0 : tensor<1x?x?xf32> {"func.input_symbol_0" = "c1", "func.input_symbol_1" = "s0", "func.input_symbol_2" = "s0"}) -> tensor<1x?x?xf32>  attributes {"entrance"}{
    %1 = "llh.add"(%0, %0) : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    %2 = "llh.constant"() {"value" = dense<3.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    %3 = "llh.add"(%1, %2) : (tensor<1x?x?xf32>, tensor<1xf32>) -> tensor<1x?x?xf32>
    func.return %3 : tensor<1x?x?xf32>
  }
}
```
以上示例IR是一个简单的加法运算。使用llc-opt进行优化，其中经过100多个下降与优化Pass将其下降到LLVM IR，如下图所示：
```mlir
module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr) attributes {entrance, symbol_int_arg_nums = 0 : i64} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.getelementptr inbounds %arg0[8] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %6 = llvm.load %5 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[7] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[5] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %arg0[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %12 = llvm.load %11 : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %arg0[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr
    %15 = llvm.load %6 : !llvm.ptr -> i64
    %16 = llvm.getelementptr inbounds %10[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %17 = llvm.load %16 : !llvm.ptr -> i64
    %18 = llvm.load %10 : !llvm.ptr -> i64
    %19 = llvm.getelementptr inbounds %12[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %20 = llvm.load %19 : !llvm.ptr -> i64
    %21 = llvm.getelementptr inbounds %12[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.mul %22, %22 : i64
    %24 = llvm.mul %23, %4 : i64
    %25 = llvm.getelementptr %1[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.add %26, %0 : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.sub %0, %4 : i64
    %31 = llvm.add %29, %30 : i64
    %32 = llvm.urem %31, %0 : i64
    %33 = llvm.sub %31, %32 : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%35: i64):  // 2 preds: ^bb0, ^bb5
    %36 = llvm.icmp "slt" %35, %22 : i64
    llvm.cond_br %36, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%37: i64):  // 2 preds: ^bb2, ^bb4
    %38 = llvm.icmp "slt" %37, %20 : i64
    llvm.cond_br %38, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %39 = llvm.mul %18, %3 : i64
    %40 = llvm.mul %35, %17 : i64
    %41 = llvm.add %39, %40 : i64
    %42 = llvm.add %41, %37 : i64
    %43 = llvm.getelementptr %14[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %44 = llvm.load %43 : !llvm.ptr -> f32
    %45 = llvm.fadd %44, %44 : f32
    %46 = llvm.fadd %45, %2 : f32
    %47 = llvm.mul %23, %3 : i64
    %48 = llvm.mul %35, %22 : i64
    %49 = llvm.add %47, %48 : i64
    %50 = llvm.add %49, %37 : i64
    %51 = llvm.getelementptr %34[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %46, %51 : f32, !llvm.ptr
    %52 = llvm.add %37, %4 : i64
    llvm.br ^bb3(%52 : i64)
  ^bb5:  // pred: ^bb3
    %53 = llvm.add %35, %4 : i64
    llvm.br ^bb1(%53 : i64)
  ^bb6:  // pred: ^bb1
    %54 = llvm.mul %4, %4 : i64
    %55 = llvm.mul %54, %22 : i64
    %56 = llvm.mul %55, %22 : i64
    %57 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.mul %56, %58 : i64
    %60 = llvm.getelementptr %8[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%60, %34, %59) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @free(%28) : (!llvm.ptr) -> ()
    llvm.return
  }
  module @__symbol__ {
  }
}
```

# llc-translate
之后使用命令：llc-translate llc_module.opted.mlir -mlir-to-llvmir -o llc_module.ll
就会将上述的MLIR IR 翻译为 LLVM IR
```mlir
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define void @main(ptr %0) {
  %2 = getelementptr inbounds ptr, ptr %0, i32 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds ptr, ptr %0, i32 7
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds ptr, ptr %0, i32 5
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr inbounds ptr, ptr %0, i32 4
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds ptr, ptr %0, i32 2
  %11 = load ptr, ptr %10, align 8
  %12 = load i64, ptr %3, align 4
  %13 = getelementptr inbounds i64, ptr %7, i32 1
  %14 = load i64, ptr %13, align 4
  %15 = load i64, ptr %7, align 4
  %16 = getelementptr inbounds i64, ptr %9, i32 2
  %17 = load i64, ptr %16, align 4
  %18 = getelementptr inbounds i64, ptr %9, i32 1
  %19 = load i64, ptr %18, align 4
  %20 = mul i64 %19, %19
  %21 = mul i64 %20, 1
  %22 = getelementptr float, ptr null, i64 %21
  %23 = ptrtoint ptr %22 to i64
  %24 = add i64 %23, 64
  %25 = call ptr @malloc(i64 %24)
  %26 = ptrtoint ptr %25 to i64
  %27 = add i64 %26, 63
  %28 = urem i64 %27, 64
  %29 = sub i64 %27, %28
  %30 = inttoptr i64 %29 to ptr
  br label %31

31:                                               ; preds = %53, %1
  %32 = phi i64 [ %54, %53 ], [ 0, %1 ]
  %33 = icmp slt i64 %32, %19
  br i1 %33, label %34, label %55

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %52, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, %17
  br i1 %37, label %38, label %53

38:                                               ; preds = %35
  %39 = mul i64 %15, 0
  %40 = mul i64 %32, %14
  %41 = add i64 %39, %40
  %42 = add i64 %41, %36
  %43 = getelementptr float, ptr %11, i64 %42
  %44 = load float, ptr %43, align 4
  %45 = fadd float %44, %44
  %46 = fadd float %45, 3.000000e+00
  %47 = mul i64 %20, 0
  %48 = mul i64 %32, %19
  %49 = add i64 %47, %48
  %50 = add i64 %49, %36
  %51 = getelementptr float, ptr %30, i64 %50
  store float %46, ptr %51, align 4
  %52 = add i64 %36, 1
  br label %35

53:                                               ; preds = %35
  %54 = add i64 %32, 1
  br label %31

55:                                               ; preds = %31
  %56 = mul i64 1, %19
  %57 = mul i64 %56, %19
  %58 = mul i64 %57, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %59 = getelementptr float, ptr %5, i64 %12
  call void @llvm.memcpy.p0.p0.i64(ptr %59, ptr %30, i64 %58, i1 false)
  call void @free(ptr %25)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```
# opt
opt 是llvm 项目的优化工具，
使用opt llc_module.ll -o llc_module.opted.ll -O3 -S --march=x86-64 --mcpu=tigerlake --mtriple=x86_64-linux-gnu 获得优化之后的llvm ir，之后将其翻译为bitcode。

opt llc_module.opted.ll -o llc_module.bc --march=x86-64 --mcpu=tigerlake --mtriple=x86_64-linux-gnu

# llc
使用以下命令将bitcode文件编译为obj文件
llc llc_module.bc -o llc_module.o -filetype=obj -relocation-model=pic --march=x86-64 --mcpu=tigerlake --mtriple=x86_64-linux-gnu

最后将obj文件链接相关的库文件，将模型编译为一个so文件，
g++ llc_module.o -o llc_module.so -shared -fPIC -lmlir_c_runner_utils -L/lib -L/llvm/lib

