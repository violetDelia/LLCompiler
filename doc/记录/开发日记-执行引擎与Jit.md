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
  Engine(std::unique_ptr<llvm::orc::LLJIT> engine);

  void debug_info();

  int run(std::vector<Tensor*> &inputs, std::vector<Tensor*>& outs);

  std::unique_ptr<llvm::orc::LLJIT> engine;
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

## 统一执行的参数

为了将不规则的tensor信息传递同意的形式，添加一个pass将参数统一改为void ** 的形式传入，这样无论输入的形状和类型，都可以 viod （void **） 的方式对其进行调用。

在实际运行时，找到编译好的入口函数，将输入输出以统一的格式封装起来，传到到编译好的模型中。

```cpp
int Engine::run(std::vector<Tensor*>& inputs, std::vector<Tensor*>& outs) {
  auto maybe_func = engine->lookup("main");
  CHECK(llc::GLOBAL, maybe_func) << "count not find function!";
  auto& func = maybe_func.get();
  auto in = inputs[0];
  auto out = outs[0];
  std::vector<void*> params;
  for (auto tensor : inputs) {
    params.push_back(static_cast<void*>(tensor->base));
    params.push_back(static_cast<void*>(tensor->data));
    params.push_back(static_cast<void*>(&tensor->offset));
    params.push_back(static_cast<void*>(tensor->size.data()));
    params.push_back(static_cast<void*>(tensor->stride.data()));
  }
  for (auto tensor : outs) {
    params.push_back(static_cast<void*>(tensor->base));
    params.push_back(static_cast<void*>(tensor->data));
    params.push_back(static_cast<void*>(&tensor->offset));
    params.push_back(static_cast<void*>(tensor->size.data()));
    params.push_back(static_cast<void*>(tensor->stride.data()));
  }
  auto run = func.toPtr<void(void**)>(); // 入口函数
  run(static_cast<void**>(params.data()));
  return 0;
}
```

之前的IR：

```
llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) attributes {entrance} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
```

转变后的为：

```
llvm.func @main(%arg0: !llvm.ptr) attributes {entrance} {
    %0 = llvm.mlir.constant(9 : index) : i64
    %1 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %5 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
    %6 = llvm.mlir.constant(7 : index) : i64
    %7 = llvm.getelementptr inbounds %arg0[%6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.mlir.constant(6 : index) : i64
    %10 = llvm.getelementptr inbounds %arg0[%9] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
    %12 = llvm.mlir.constant(5 : index) : i64
    %13 = llvm.getelementptr inbounds %arg0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr
    %15 = llvm.mlir.constant(4 : index) : i64
    %16 = llvm.getelementptr inbounds %arg0[%15] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
    %18 = llvm.mlir.constant(3 : index) : i64
    %19 = llvm.getelementptr inbounds %arg0[%18] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.ptr
    %21 = llvm.mlir.constant(2 : index) : i64
    %22 = llvm.getelementptr inbounds %arg0[%21] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %23 = llvm.load %22 : !llvm.ptr -> !llvm.ptr
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.getelementptr inbounds %arg0[%24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %26 = llvm.load %25 : !llvm.ptr -> !llvm.ptr
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.getelementptr inbounds %arg0[%27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %29 = llvm.load %28 : !llvm.ptr -> !llvm.ptr
    %30 = llvm.mlir.constant(3 : index) : i64
    %31 = llvm.getelementptr inbounds %2[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %32 = llvm.load %31 : !llvm.ptr -> i64
    %33 = llvm.mlir.constant(2 : index) : i64
    %34 = llvm.getelementptr inbounds %2[%33] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %35 = llvm.load %34 : !llvm.ptr -> i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.getelementptr inbounds %2[%36] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %38 = llvm.load %37 : !llvm.ptr -> i64
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.getelementptr inbounds %2[%39] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %41 = llvm.load %40 : !llvm.ptr -> i64
    %42 = llvm.mlir.constant(3 : index) : i64
    %43 = llvm.getelementptr inbounds %5[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %44 = llvm.load %43 : !llvm.ptr -> i64
    %45 = llvm.mlir.constant(2 : index) : i64
    %46 = llvm.getelementptr inbounds %5[%45] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %47 = llvm.load %46 : !llvm.ptr -> i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.getelementptr inbounds %5[%48] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %50 = llvm.load %49 : !llvm.ptr -> i64
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.getelementptr inbounds %5[%51] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %53 = llvm.load %52 : !llvm.ptr -> i64
    %54 = llvm.mlir.constant(0 : index) : i64
    %55 = llvm.getelementptr inbounds %8[%54] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %56 = llvm.load %55 : !llvm.ptr -> i64
    %57 = llvm.mlir.constant(3 : index) : i64
    %58 = llvm.getelementptr inbounds %17[%57] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %59 = llvm.load %58 : !llvm.ptr -> i64
    %60 = llvm.mlir.constant(2 : index) : i64
    %61 = llvm.getelementptr inbounds %17[%60] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %62 = llvm.load %61 : !llvm.ptr -> i64
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.getelementptr inbounds %17[%63] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %65 = llvm.load %64 : !llvm.ptr -> i64
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.getelementptr inbounds %17[%66] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %68 = llvm.load %67 : !llvm.ptr -> i64
    %69 = llvm.mlir.constant(3 : index) : i64
    %70 = llvm.getelementptr inbounds %20[%69] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %71 = llvm.load %70 : !llvm.ptr -> i64
    %72 = llvm.mlir.constant(2 : index) : i64
    %73 = llvm.getelementptr inbounds %20[%72] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %74 = llvm.load %73 : !llvm.ptr -> i64
    %75 = llvm.mlir.constant(1 : index) : i64
    %76 = llvm.getelementptr inbounds %20[%75] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %77 = llvm.load %76 : !llvm.ptr -> i64
    %78 = llvm.mlir.constant(0 : index) : i64
    %79 = llvm.getelementptr inbounds %20[%78] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %80 = llvm.load %79 : !llvm.ptr -> i64
    %81 = llvm.mlir.constant(0 : index) : i64
    %82 = llvm.getelementptr inbounds %23[%81] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %83 = llvm.load %82 : !llvm.ptr -> i64
```

## 在python和C++中进行Tensor的传递

为了将python的tensor张量传递到C++中执行，在pybind 中指定了Tensor和numpy的转换规则：

```cpp
pybind11::class_<Tensor>(entrance, "Tensor", py::buffer_protocol())
      .def_readwrite("data", &Tensor::data)
      .def(pybind11::init<size_t, size_t, size_t, size_t, std::vector<size_t> &,
                          std::vector<size_t> &>())
      .def_readwrite("data", &Tensor::data)
      .def_readwrite("base", &Tensor::base)
      .def_readwrite("offset", &Tensor::offset)
      .def_readwrite("size", &Tensor::size)
      .def_readwrite("stride", &Tensor::stride)
      .def("print", &Tensor::print)
      .def_buffer([](Tensor &self) -> py::buffer_info {
        return py::buffer_info(self.base, get_itemsize(self.type),
                               get_format(self.type), self.size.size(),
                               self.size, get_stride_in(&self));
      })
      .def("to_numpy", [](Tensor *self) {
        auto bufer = py::buffer_info(self->base, get_itemsize(self->type),
                                     get_format(self->type), self->size.size(),
                                     self->size, get_stride_in(self));
        return py::array(bufer);
      });
```

之后再python端封装一个执行器，用来将数据结构进行转换，以及推导输出的tensor信息。

```python
def run(self, *args) -> Any:
        inputs = self.trans_to_tensor(*args)  # 将torch.Tensor 转变为C++定义的Tensor
        outputs = self.gen_outs_call(*args)  # 推导输出的tensor信息，并分配好内存
        outputs_ = self.trans_to_tensor(
            *outputs
        )  # 输出的torch.Tensor 转变为C++定义的Tensor
        self.engine.run(inputs, outputs_)  # 调用执行函数
        return outputs

```

## 执行

现在，编译器已经具备执行模型的功能了，定义一个简单的模型：
```python
class ElementaryArithmetic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x1 = x.reshape(x.shape[3], x.shape[2], x.shape[0], x.shape[1])
        x1 = x1 + x1
        x1 = x - 2
        x1 = x1 * 2
        x1 = x1 / 2
        x2 = x.reshape(x.shape[3], x.shape[0], x.shape[2], x.shape[1])
        x2 = x2 + x2
        x2 = x2.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = x2 - x1
        return x
```

```python
if __name__ == "__main__":
    # run_model_dict(module_dict)
    model = ElementaryArithmetic()
    input = torch.ones(2, 2, 2, 5)
    compiler = LLC.LLCompiler(
        mode="inference",
        symbol_infer=True,
    )
    opt_model: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
        model=model,
        backend=compiler,
        dynamic=False,
        fullgraph=True,
    )
    print("llcompiler")
    print(opt_model(input))
    print("torch")
    print(model(input))
'''
llcompiler
tensor([[[[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]],

         [[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]]],


        [[[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]],

         [[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]]]])
torch
tensor([[[[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]],

         [[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]]],


        [[[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]],

         [[3., 3., 3., 3., 3.],
          [3., 3., 3., 3., 3.]]]])
'''
```
现在编译器已经具备了简单运算和广播的能力，但是其性能与高性能算子大约有5~10倍的差距，之后会逐步的介绍编译器的优化和模型的优化，来提高编译器的运行效率。
```python
模型:  Add , 模式:  training
llcompiler_run_time : time is 1.957s
torch_run_time : time is 0.015s
模型:  Div , 模式:  training
llcompiler_run_time : time is 0.106s
torch_run_time : time is 0.015s
Div  in  training  is incorrect!
模型:  Sub , 模式:  training
llcompiler_run_time : time is 0.104s
torch_run_time : time is 0.015s
模型:  Mul , 模式:  training
llcompiler_run_time : time is 0.106s
torch_run_time : time is 0.013s
模型:  ElementaryArithmetic , 模式:  training
llcompiler_run_time : time is 0.196s
torch_run_time : time is 0.041s
模型:  Add , 模式:  inference
llcompiler_run_time : time is 0.100s
torch_run_time : time is 0.016s
模型:  Div , 模式:  inference
llcompiler_run_time : time is 0.102s
torch_run_time : time is 0.017s
Div  in  inference  is incorrect!
模型:  Sub , 模式:  inference
llcompiler_run_time : time is 0.102s
torch_run_time : time is 0.016s
模型:  Mul , 模式:  inference
llcompiler_run_time : time is 0.103s
torch_run_time : time is 0.014s
模型:  ElementaryArithmetic , 模式:  inference
llcompiler_run_time : time is 0.190s
torch_run_time : time is 0.046s

```
