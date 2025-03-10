# 前言

最近为了调试attention的计算苦不堪言，为了能够方便的查看到底到底是计算到哪里出了问题，借助MLIR社区已经实现的功能，实现了一个用于debug的PrintOp。

定义如下：

```
def LLH_PrintOp   : Op<LLH_Dialect, "print", [MemoryEffects<[MemWrite,MemRead]>]>{
    let arguments = (ins    LLH_AnyType:$input,
                            LLH_StringAttr:$prefix_description);
}

```

其中 `MemoryEffects<[MemWrite,MemRead]` 的接口是标记printOp会对内存进行读写，防止优化直接把想要打印的内存给优化掉。

这个Op最终会在lowing到llvm ir 的时候lowing成func.func，调用打印内存的函数。因此它的输入可以是任意值。

为了能够让他在内存分析阶段能够顺利lowing下去，还需要实现它的bufferize接口，如下所示：

```
struct PrintOpInterface
    : public BufferizableOpInterface::ExternalModel<PrintOpInterface,
                                                    llh::PrintOp> {
 public:
  bool bufferizesToAllocation(Operation *op, ::mlir::Value value) const {
    return false;
  };
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }
  bool bufferizesToElementwiseAccess(
      Operation *op, const ::mlir::bufferization::AnalysisState &state,
      ArrayRef<OpOperand *> opOperands) const {
    return true;
  };
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  };
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto print = cast<llh::PrintOp>(op);
    auto input = print.getInput();
    auto input_type = input.getType();
    if (auto tensor_type =
            llvm::dyn_cast_or_null<mlir::TensorType>(input_type)) {
      auto memref_type = bufferization::getBufferType(input, options);
      auto to_memref = rewriter.create<bufferization::ToMemrefOp>(
          print->getLoc(), *memref_type, input, true);
      replaceOpWithNewBufferizedOp<llh::PrintOp>(
          rewriter, op, to_memref->getResult(0), print.getPrefixDescription());
      return llvm::success();
    }
    return llvm::failure();
  }
};
```

因为目前还只是在cpu上运行，因此可以直接复用mlir_runner_utils.so里面的函数，只要编译阶段链接它就好了。同时要将这两个库添加到lit套件的检测文件中：

```
tools = [
    'llc-opt',
    'llc-translate',
    ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
    add_runtime("mlir_runner_utils"),
    add_runtime("mlir_c_runner_utils"),
]
```

这样，就可以在lit测试中借助mlir-runner 直接查看计算图的输出结果是否符合预期了。

如下所示：

```
// RUN: llc-opt %s -transform-pipeline | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

module{
    func.func @main() -> () attributes {entrance} {
        %const = "llh.constant"() <{value = dense<-1.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
        %0 = "llh.abs"(%const) : (tensor<2xf32>) -> tensor<2xf32>
        // CHECK: abs
        // CHECK: Unranked Memref
        // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data = 
        // CHECK-NEXT: [1,  1]
        "llh.print"(%0) <{prefix_description = "abs"}>: (tensor<2xf32>) -> ()
        return 
  }
}
```
