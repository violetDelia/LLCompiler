# 引言

经过在mlir一系列的Pass优化，mlir已经lowing到了llvm dialect 上，那么接下来需要将其翻译成llvm并且执行。这样整个模型的编译工作就基本完成了，之后的事情就是添加各种优化pass，让程序执行的更快，更节省硬件资源。

## MLIR To LLVM

    现在我们经过mlir的一系列pass获得的是一张llvm dialect'的IR图：

```
module attributes {builtin.gloabal_layout = "NCHW"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {entrance} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.mlir.constant(4 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %8 = llvm.getelementptr %1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.add %9, %0 : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.sub %0, %5 : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.urem %14, %0  : i64
    %16 = llvm.sub %14, %15 : i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %18 = llvm.insertvalue %11, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %17, %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %6, %19[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %2, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %5, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %4, %22[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %4, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %4, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %5, %25[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%27: i64):  // 2 preds: ^bb0, ^bb2
    %28 = llvm.icmp "slt" %27, %3 : i64
    llvm.cond_br %28, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %29 = llvm.urem %27, %4  : i64
    %30 = llvm.udiv %27, %4  : i64
    %31 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.mul %30, %arg6 : i64
    %33 = llvm.mul %arg7, %6 : i64
    %34 = llvm.add %32, %33 : i64
    %35 = llvm.mul %29, %arg8 : i64
    %36 = llvm.add %34, %35 : i64
    %37 = llvm.getelementptr %31[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.fadd %38, %38  : f32
    %40 = llvm.mul %30, %4 : i64
    %41 = llvm.mul %6, %4 : i64
    %42 = llvm.add %40, %41 : i64
    %43 = llvm.add %42, %29 : i64
    %44 = llvm.getelementptr %17[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %39, %44 : f32, !llvm.ptr
    %45 = llvm.add %27, %5 : i64
    llvm.br ^bb1(%45 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return %26 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
```

调用接口实现将其翻译为llvm的IR：

```cpp
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    auto llvm_module = mlir::translateModuleToLLVMIR(module.get(), *llvm_context);
```

然后设置机器平台对其进行优化：

```cpp
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    CHECK(llc::GLOBAL, tmBuilderOrError)
        << "Could not create JITTargetMachineBuilder\n";
    auto tmOrError = tmBuilderOrError->createTargetMachine();
    CHECK(llc::GLOBAL, tmOrError) << "Could not create TargetMachine\n";
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvm_module.get(),
                                                            tmOrError.get().get());
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/options.opt_level, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
```

优化之后的IR图已经如下所示。

```
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #0

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: readwrite)
define { ptr, ptr, i64, [3 x i64], [3 x i64] } @main(ptr nocapture readnone %0, ptr nocapture readonly %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8) local_unnamed_addr #1 {
  %10 = tail call dereferenceable_or_null(96) ptr @malloc(i64 96)
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = and i64 %12, -64
  %14 = inttoptr i64 %13 to ptr
  %15 = getelementptr float, ptr %1, i64 %2
  %16 = load float, ptr %15, align 4
  %17 = fadd float %16, %16
  store float %17, ptr %14, align 64
  %18 = getelementptr float, ptr %15, i64 %8
  %19 = load float, ptr %18, align 4
  %20 = fadd float %19, %19
  %21 = getelementptr i8, ptr %14, i64 4
  store float %20, ptr %21, align 4
  %22 = shl i64 %8, 1
  %23 = getelementptr float, ptr %15, i64 %22
  %24 = load float, ptr %23, align 4
  %25 = fadd float %24, %24
  %26 = getelementptr i8, ptr %14, i64 8
  store float %25, ptr %26, align 8
  %27 = mul i64 %8, 3
  %28 = getelementptr float, ptr %15, i64 %27
  %29 = load float, ptr %28, align 4
  %30 = fadd float %29, %29
  %31 = getelementptr i8, ptr %14, i64 12
  store float %30, ptr %31, align 4
  %32 = getelementptr float, ptr %15, i64 %6
  %33 = load float, ptr %32, align 4
  %34 = fadd float %33, %33
  %35 = getelementptr i8, ptr %14, i64 16
  store float %34, ptr %35, align 16
  %36 = getelementptr float, ptr %32, i64 %8
  %37 = load float, ptr %36, align 4
  %38 = fadd float %37, %37
  %39 = getelementptr i8, ptr %14, i64 20
  store float %38, ptr %39, align 4
  %40 = getelementptr float, ptr %32, i64 %22
  %41 = load float, ptr %40, align 4
  %42 = fadd float %41, %41
  %43 = getelementptr i8, ptr %14, i64 24
  store float %42, ptr %43, align 8
  %44 = getelementptr float, ptr %32, i64 %27
  %45 = load float, ptr %44, align 4
  %46 = fadd float %45, %45
  %47 = getelementptr i8, ptr %14, i64 28
  store float %46, ptr %47, align 4
  %48 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %10, 0
  %49 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %48, ptr %14, 1
  %50 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, i64 0, 2
  %51 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %50, i64 2, 3, 0
  %52 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %51, i64 1, 3, 1
  %53 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %52, i64 4, 3, 2
  %54 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %53, i64 4, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %54, i64 4, 4, 1
  %56 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, i64 1, 4, 2
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %56
}

attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

## Jit执行引擎

现在我们拿到了llvm的bitcode形式的IR图，下一步就是执行它：
我们采用LLVM最新的执行引擎，LLJIT，而且这个引擎可以很方便的加载动态库。
加载已经翻译好的LLVM IR图：

```cpp
    auto maybe_jit = llvm::orc::LLJITBuilder().setNumCompileThreads(8).create();
    CHECK(llc::GLOBAL, maybe_jit) << "Failed to create JIT";
    auto& jit = maybe_jit.get();
    auto error = jit->addIRModule(llvm::orc::ThreadSafeModule(
        std::move(llvm_module), std::move(llvm_context)));
    CHECK(llc::GLOBAL, !error) << "Failed to add module!";
```

然后将其封装一下，方便再python端调用：

```cpp
extern "C" struct Engine {
  Engine(llvm::orc::LLJIT* engine);

  void debug_info();

  std::vector<Tensor*> run(std::vector<Tensor*>& inputs);

  llvm::orc::LLJIT* engine;
};
```

其中Tensor的数据结构是用来描述Tensor张量信息的，它的结构很像mlir的memref，很适合当作传入参数来调用定义如下：
```
extern "C" struct Tensor {
  Tensor();
  Tensor(size_t data_ptr, size_t base_ptr, size_t type, size_t offset,
         std::vector<size_t>& size, std::vector<size_t>& stride);
  void print();

  void* data;
  void* base;
  Type type;
  size_t offset;
  std::vector<size_t> size;
  std::vector<size_t> stride;
};
```
