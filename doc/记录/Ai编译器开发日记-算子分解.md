# 算子分解

算子分解时图优化中十分重要的环节。它指将复杂的大算子拆解为简单的"元"算子组合。

一方面：将大算子拆解为小算子组合，可以为我们找到更多的优化空间，举一个典型的例子：

```
x = tensor(100x100)
// 分解
LeakyRlue(x, k) + 3 --> maximum(x,0) * k + 3
// 常量折叠
maximum(x,0) * k + 3 --> maximum(x, 3/k) * k
```

经过上述的变换，将10000次加的操作变为了一次除法操作，减少了运算量。

另一方面，算子分解可以看作是代码生成的第一步。框架支持的算子有好几百个，为所有的算子去做代码生成工作量巨大的事情。如果将好几百个算子压缩为几十中元算子，然后去做代码生成极大的减少了工作量。而且基本不会产生副作用。

## 示例

最典型的需要分解的算子有batchnorm、layernorm、gloabal_maxpool 等。

以batchnorm 为例：

模型定义

```
class Decompose_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(224, 100)
        self.batch1 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 10)
        self.batch2 = nn.BatchNorm1d(10)
        self.rule = nn.ReLU()
        self.flaten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.batch1(x)
        x = self.rule(x)
        x = self.linear2(x)
        x = self.batch2(x)
        x = self.rule(x)
        x = self.flaten(x)
        return x
```

从框架拿到的原始图：

```
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c100"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c224"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @main(%arg0: tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c100, value = 100 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c10, value = 10 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.weight"() <{weight_file = "linear1.weight.npy"}> : () -> tensor<100x224xf32, #llh.encoding<shapes = @c100, @c224>>
    %5 = "llh.weight"() <{weight_file = "linear1.bias.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %6 = "llh.weight"() <{weight_file = "batch1.weight.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %7 = "llh.weight"() <{weight_file = "batch1.bias.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %8 = "llh.weight"() <{weight_file = "linear2.weight.npy"}> : () -> tensor<10x100xf32, #llh.encoding<shapes = @c10, @c100>>
    %9 = "llh.weight"() <{weight_file = "linear2.bias.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %10 = "llh.weight"() <{weight_file = "batch2.weight.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %11 = "llh.weight"() <{weight_file = "batch2.bias.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %12 = "llh.weight"() <{weight_file = "batch1.running_mean.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %13 = "llh.weight"() <{weight_file = "batch1.running_var.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %14 = "llh.weight"() <{weight_file = "batch1.num_batches_tracked.npy"}> : () -> tensor<1xi64, #llh.encoding<shapes = @c1>>
    %15 = "llh.weight"() <{weight_file = "batch2.running_mean.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %16 = "llh.weight"() <{weight_file = "batch2.running_var.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %17 = "llh.weight"() <{weight_file = "batch2.num_batches_tracked.npy"}> : () -> tensor<1xi64, #llh.encoding<shapes = @c1>>
    %18 = "llh.transpose"(%4) <{perms = array<i64: 1, 0>}> : (tensor<100x224xf32, #llh.encoding<shapes = @c100, @c224>>) -> tensor<224x100xf32, #llh.encoding<shapes = @c224, @c100>>
    %19 = "llh.matmul"(%arg0, %18) : (tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>>, tensor<224x100xf32, #llh.encoding<shapes = @c224, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %20 = "llh.reshape"(%5, %3, %0) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @c1, @c100>>
    %21 = "llh.dim"(%19, %2) <{symbol = @s0}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64) -> i64
    %22 = "llh.broadcast_to"(%20, %21, %0) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @c1, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %23 = "llh.add"(%19, %22) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %24 = "llh.batch_norm"(%23, %6, %7, %12, %13) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<100xf32, #llh.encoding<shapes = @c100>>, tensor<100xf32, #llh.encoding<shapes = @c100>>, tensor<100xf32, #llh.encoding<shapes = @c100>>, tensor<100xf32, #llh.encoding<shapes = @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %25 = "llh.relu"(%24) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %26 = "llh.transpose"(%8) <{perms = array<i64: 1, 0>}> : (tensor<10x100xf32, #llh.encoding<shapes = @c10, @c100>>) -> tensor<100x10xf32, #llh.encoding<shapes = @c100, @c10>>
    %27 = "llh.matmul"(%25, %26) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<100x10xf32, #llh.encoding<shapes = @c100, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %28 = "llh.reshape"(%9, %3, %1) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    %29 = "llh.dim"(%27, %2) <{symbol = @s0}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64) -> i64
    %30 = "llh.broadcast_to"(%28, %29, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %31 = "llh.add"(%27, %30) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %32 = "llh.batch_norm"(%31, %10, %11, %15, %16) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<10xf32, #llh.encoding<shapes = @c10>>, tensor<10xf32, #llh.encoding<shapes = @c10>>, tensor<10xf32, #llh.encoding<shapes = @c10>>, tensor<10xf32, #llh.encoding<shapes = @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %33 = "llh.relu"(%32) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %34 = "llh.dim"(%33, %2) <{symbol = @s0}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64) -> i64
    %35 = "llh.reshape"(%33, %34, %1) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    return %35 : tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
  }
  module @__symbol__ {
  }
}



```

经过算子分解后：

```
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c100"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c224"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @main(%arg0: tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>> attributes {entrance} {
    %0 = "llh.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %1 = "llh.constant"() <{symbol = @c100, value = 100 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c10, value = 10 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %5 = "llh.weight"() <{weight_file = "linear1.weight.npy"}> : () -> tensor<100x224xf32, #llh.encoding<shapes = @c100, @c224>>
    %6 = "llh.weight"() <{weight_file = "linear1.bias.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %7 = "llh.weight"() <{weight_file = "batch1.weight.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %8 = "llh.weight"() <{weight_file = "batch1.bias.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %9 = "llh.weight"() <{weight_file = "linear2.weight.npy"}> : () -> tensor<10x100xf32, #llh.encoding<shapes = @c10, @c100>>
    %10 = "llh.weight"() <{weight_file = "linear2.bias.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %11 = "llh.weight"() <{weight_file = "batch2.weight.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %12 = "llh.weight"() <{weight_file = "batch2.bias.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %13 = "llh.weight"() <{weight_file = "batch1.running_mean.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %14 = "llh.weight"() <{weight_file = "batch1.running_var.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %15 = "llh.weight"() <{weight_file = "batch1.num_batches_tracked.npy"}> : () -> tensor<1xi64, #llh.encoding<shapes = @c1>>
    %16 = "llh.weight"() <{weight_file = "batch2.running_mean.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %17 = "llh.weight"() <{weight_file = "batch2.running_var.npy"}> : () -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %18 = "llh.weight"() <{weight_file = "batch2.num_batches_tracked.npy"}> : () -> tensor<1xi64, #llh.encoding<shapes = @c1>>
    %19 = "llh.transpose"(%5) <{perms = array<i64: 1, 0>}> : (tensor<100x224xf32, #llh.encoding<shapes = @c100, @c224>>) -> tensor<224x100xf32, #llh.encoding<shapes = @c224, @c100>>
    %20 = "llh.matmul"(%arg0, %19) : (tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>>, tensor<224x100xf32, #llh.encoding<shapes = @c224, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %21 = "llh.reshape"(%6, %4, %1) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @c1, @c100>>
    %22 = "llh.dim"(%20, %3) <{symbol = @s0}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64) -> i64
    %23 = "llh.broadcast_to"(%21, %22, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @c1, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %24 = "llh.add"(%20, %23) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %25 = "llh.add"(%14, %0) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %26 = "llh.sqrt"(%25) : (tensor<100xf32, #llh.encoding<shapes = @c100>>) -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %27 = "llh.dim"(%24, %4) <{symbol = @c100}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64) -> i64
    %28 = "llh.dim"(%24, %3) <{symbol = @s0}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64) -> i64
    %29 = "llh.reshape"(%13, %4, %27) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %30 = "llh.broadcast_to"(%29, %28, %27) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %31 = "llh.reshape"(%7, %4, %27) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %32 = "llh.broadcast_to"(%31, %28, %27) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %33 = "llh.reshape"(%8, %4, %27) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %34 = "llh.broadcast_to"(%33, %28, %27) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %35 = "llh.reshape"(%26, %4, %27) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %36 = "llh.broadcast_to"(%35, %28, %27) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %37 = "llh.sub"(%24, %30) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %38 = "llh.mul"(%37, %32) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %39 = "llh.div"(%38, %36) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %40 = "llh.add"(%39, %34) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %41 = "llh.relu"(%40) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %42 = "llh.transpose"(%9) <{perms = array<i64: 1, 0>}> : (tensor<10x100xf32, #llh.encoding<shapes = @c10, @c100>>) -> tensor<100x10xf32, #llh.encoding<shapes = @c100, @c10>>
    %43 = "llh.matmul"(%41, %42) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<100x10xf32, #llh.encoding<shapes = @c100, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %44 = "llh.reshape"(%10, %4, %2) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    %45 = "llh.dim"(%43, %3) <{symbol = @s0}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64) -> i64
    %46 = "llh.broadcast_to"(%44, %45, %2) <{cast_dims = array<i64: 0>}> : (tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %47 = "llh.add"(%43, %46) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %48 = "llh.add"(%17, %0) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %49 = "llh.sqrt"(%48) : (tensor<10xf32, #llh.encoding<shapes = @c10>>) -> tensor<10xf32, #llh.encoding<shapes = @c10>>
    %50 = "llh.dim"(%47, %4) <{symbol = @c10}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64) -> i64
    %51 = "llh.dim"(%47, %3) <{symbol = @s0}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64) -> i64
    %52 = "llh.reshape"(%16, %4, %50) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %53 = "llh.broadcast_to"(%52, %51, %50) <{cast_dims = array<i64: 0>}> : (tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %54 = "llh.reshape"(%11, %4, %50) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %55 = "llh.broadcast_to"(%54, %51, %50) <{cast_dims = array<i64: 0>}> : (tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %56 = "llh.reshape"(%12, %4, %50) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %57 = "llh.broadcast_to"(%56, %51, %50) <{cast_dims = array<i64: 0>}> : (tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %58 = "llh.reshape"(%49, %4, %50) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %59 = "llh.broadcast_to"(%58, %51, %50) <{cast_dims = array<i64: 0>}> : (tensor<1x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %60 = "llh.sub"(%47, %53) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %61 = "llh.mul"(%60, %55) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %62 = "llh.div"(%61, %59) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %63 = "llh.add"(%62, %57) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %64 = "llh.relu"(%63) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %65 = "llh.dim"(%64, %3) <{symbol = @s0}> : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64) -> i64
    %66 = "llh.reshape"(%64, %65, %2) : (tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>, i64, i64) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    return %66 : tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
  }
  module @__symbol__ {
  }
}
```

时间记录：

torch eager：0.042s

torch compiler：0.002s

llcompiler：0.007s
