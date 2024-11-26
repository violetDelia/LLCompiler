# 内存复用

内存复用是一种内存优化手段，可以有效的降低运行模型时设备的内存使用量。它的基本思想是通过分析张量的生存周期，确定那些张量在计算种不在需要，这样就可以将它所占用的内存释放出来，给其他的张量使用。

另一方面，因为编译的IR是基于SSA语义的，即每一个计算生成的张量都代表一个新的张量。一次如果不进行内存复用，在编译之后运行会反复进行内存分配，极大的增加了运行时的消耗。

常见的内存复用方法有两种：内存置换和内存共享

相关链接：[Bufferization ODM](https://mlir.llvm.org/OpenMeetings/2022-01-13-One-Shot-Bufferization.pdf)

## 内存置换

它其实就是torch种的inplace 操作，它会为每一个生成的张量分析其依赖关系，如果这个张量只会被之后的某一个算子单独使用，那么这个张量就可以进行置换操作。举一个简单的例子：

```
x1 = tensor.alloc(100x100)
x2 = tensor.alloc(100x100)
x3 = tanh(x2, inplace = false)
x4 = tensor.add(x3,x1) 
x5 = tanh(x4, inplace = false)

```

以上示例在计算过程中产生了5个张量。经过分析发现x3，x1，x4 只被使用了一次并且没有其他的依赖关系，那么以上的运算就可以改写为：

```
x1 = tensor(100x100)
x2 = tensor(100x100)
x2 = tanh(x2, inplace = true)
x2 = tensor.add(x2,x1) 
x2 = tanh(x2, inplace = true)
```

## 内存共享

第二种内存复用的方式为内存共享，它的实现就比较复杂了，它需要分析在生命周期。如果一个张量的生命周期结束了，之后如果分配的一个与其大小一样（其实可以不一样）。那么就不分配内存，将本应该释放的内存作为新生成的张量。

这种方法的好处是上线高，可以复用内存大小不同的张量。而内存置换只能分析出大小一样且具有依赖关系的张量进行复用。

举例说明：

```
x1 = tensor.alloc(100x100)
x2 = tensor.alloc(100x100)
x3 = tensor.alloc(100x100)
x3 = tensor.add(x2,x1) 
x4 = tensor.alloc(100x100)
x4 = tensor.add(x2,x3) 
dealloc(x2)
dealloc(x3)
x5 = tensor.alloc(100x100)
x5 = tensor.add(x4,x1) 
dealloc(x1)
dealloc(x4)
dealloc(x5)
```

可以发现 在给变量%5去分配内存时，刚好之前的x3被释放了，于是可以将x3替换为x5

```
x1 = tensor.alloc(100x100)
x2 = tensor.alloc(100x100)
x3 = tensor.alloc(100x100)
x3 = tensor.add(x2,x1) 
x4 = tensor.alloc(100x100)
x4 = tensor.add(x2,x3) 
dealloc(x2)
x3 = tensor.add(x4,x1) 
dealloc(x1)
dealloc(x3)
dealloc(x4)
```

## 实现

在MLIR中的bufferzation dialect 就是专门进行内存分析的方案，当中已经实现了很多内存分析的方法。详见[Bufferization ODM](https://mlir.llvm.org/OpenMeetings/2022-01-13-One-Shot-Bufferization.pdf)。

demo模型：

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
        # x = self.linear2(x)
        # x = self.batch2(x)
        # x = self.rule(x)
        # x = self.flaten(x)
        return x
```

从框架拿到的图是这样的：

```
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c100"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c224"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @main(%arg0: tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}) -> tensor<?x100xf32> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c100, value = 100 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %3 = "llh.weight"() <{weight_file = "linear1.weight.npy"}> : () -> tensor<100x224xf32>
    %4 = "llh.weight"() <{weight_file = "linear1.bias.npy"}> : () -> tensor<100xf32>
    %5 = "llh.weight"() <{weight_file = "batch1.weight.npy"}> : () -> tensor<100xf32>
    %6 = "llh.weight"() <{weight_file = "batch1.bias.npy"}> : () -> tensor<100xf32>
    %7 = "llh.weight"() <{weight_file = "batch1.running_mean.npy"}> : () -> tensor<100xf32>
    %8 = "llh.weight"() <{weight_file = "batch1.running_var.npy"}> : () -> tensor<100xf32>
    %9 = "llh.weight"() <{weight_file = "batch1.num_batches_tracked.npy"}> : () -> tensor<1xi64>
    %10 = "llh.transpose"(%3) <{perms = array<i64: 1, 0>}> : (tensor<100x224xf32>) -> tensor<224x100xf32>
    %11 = "llh.matmul"(%arg0, %10) : (tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>>, tensor<224x100xf32>) -> tensor<?x100xf32>
    %12 = "llh.reshape"(%4, %2, %0) : (tensor<100xf32>, i64, i64) -> tensor<1x100xf32>
    %13 = "llh.dim"(%11, %1) : (tensor<?x100xf32>, i64) -> i64
    %14 = "llh.broadcast_to"(%12, %13, %0) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32>, i64, i64) -> tensor<?x100xf32>
    %15 = "llh.add"(%11, %14) : (tensor<?x100xf32>, tensor<?x100xf32>) -> tensor<?x100xf32>
    %16 = "llh.batch_norm"(%15, %5, %6, %7, %8) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<?x100xf32>, tensor<100xf32>, tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) -> tensor<?x100xf32>
    %17 = "llh.relu"(%16) : (tensor<?x100xf32>) -> tensor<?x100xf32>
    return %17 : tensor<?x100xf32>
  }
  module @__symbol__ {
  }
}
```

将BatchNorm分解后是这样的：

```
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c100"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c224"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @main(%arg0: tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>> attributes {entrance} {
    %0 = "llh.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %1 = "llh.constant"() <{symbol = @c100, value = 100 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.weight"() <{weight_file = "linear1.weight.npy"}> : () -> tensor<100x224xf32, #llh.encoding<shapes = @c100, @c224>>
    %5 = "llh.weight"() <{weight_file = "linear1.bias.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %6 = "llh.weight"() <{weight_file = "batch1.weight.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %7 = "llh.weight"() <{weight_file = "batch1.bias.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %8 = "llh.weight"() <{weight_file = "batch1.running_mean.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %9 = "llh.weight"() <{weight_file = "batch1.running_var.npy"}> : () -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %10 = "llh.weight"() <{weight_file = "batch1.num_batches_tracked.npy"}> : () -> tensor<1xi64, #llh.encoding<shapes = @c1>>
    %11 = "llh.transpose"(%4) <{perms = array<i64: 1, 0>}> : (tensor<100x224xf32, #llh.encoding<shapes = @c100, @c224>>) -> tensor<224x100xf32, #llh.encoding<shapes = @c224, @c100>>
    %12 = "llh.matmul"(%arg0, %11) : (tensor<?x224xf32, #llh.encoding<shapes = @s0, @c224>>, tensor<224x100xf32, #llh.encoding<shapes = @c224, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %13 = "llh.reshape"(%5, %3, %1) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @c1, @c100>>
    %14 = "llh.dim"(%12, %2) <{symbol = @s0}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64) -> i64
    %15 = "llh.broadcast_to"(%13, %14, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @c1, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %16 = "llh.add"(%12, %15) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %17 = "llh.broadcast_to"(%0, %1) <{cast_dims = array<i64: 0>}> : (tensor<1xf32, #llh.encoding<shapes = @c1>>, i64) -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %18 = "llh.add"(%9, %17) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, tensor<100xf32, #llh.encoding<shapes = @c100>>) -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %19 = "llh.sqrt"(%18) : (tensor<100xf32, #llh.encoding<shapes = @c100>>) -> tensor<100xf32, #llh.encoding<shapes = @c100>>
    %20 = "llh.dim"(%16, %2) <{symbol = @s0}> : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64) -> i64
    %21 = "llh.reshape"(%8, %3, %1) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %22 = "llh.broadcast_to"(%21, %20, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %23 = "llh.reshape"(%6, %3, %1) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %24 = "llh.broadcast_to"(%23, %20, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %25 = "llh.reshape"(%7, %3, %1) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %26 = "llh.broadcast_to"(%25, %20, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %27 = "llh.reshape"(%19, %3, %1) : (tensor<100xf32, #llh.encoding<shapes = @c100>>, i64, i64) -> tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %28 = "llh.broadcast_to"(%27, %20, %1) <{cast_dims = array<i64: 0>}> : (tensor<1x100xf32, #llh.encoding<shapes = @s0, @c100>>, i64, i64) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %29 = "llh.sub"(%16, %22) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %30 = "llh.mul"(%29, %24) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %31 = "llh.div"(%30, %28) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %32 = "llh.add"(%31, %26) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>, tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    %33 = "llh.relu"(%32) : (tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>) -> tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
    return %33 : tensor<?x100xf32, #llh.encoding<shapes = @s0, @c100>>
  }
  module @__symbol__ {
  }
}
```

lowing到linalg 变成这样：

```
// -----// IR Dump After ConvertTensorToLinalg (convert-tensor-to-linalg) //----- //
#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, 0)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x224xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}) -> tensor<?x100xf32> attributes {entrance} {
    %cst = arith.constant 1.00000501 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x100xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x100xf32>
    %c0 = arith.constant 0 : index
    %cst_4 = arith.constant dense<[1, 100]> : tensor<2xi64>
    %cst_5 = arith.constant dense<[]> : tensor<100xf32>
    %cst_6 = arith.constant dense<[]> : tensor<224x100xf32>
    %dim = tensor.dim %arg0, %c0 : tensor<?x224xf32>
    %0 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<?x100xf32>) -> tensor<?x100xf32>
    %2 = linalg.matmul ins(%arg0, %cst_6 : tensor<?x224xf32>, tensor<224x100xf32>) outs(%1 : tensor<?x100xf32>) -> tensor<?x100xf32>
    %reshape = tensor.reshape %cst_5(%cst_4) : (tensor<100xf32>, tensor<2xi64>) -> tensor<1x100xf32>
    %3 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%reshape : tensor<1x100xf32>) outs(%3 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %5 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%5 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.addf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %7 = bufferization.alloc_tensor() : tensor<100xf32>
    %8 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%7 : tensor<100xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<100xf32>
    %9 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1x100xf32>) outs(%9 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %11 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<1x100xf32>) outs(%11 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %13 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1x100xf32>) outs(%13 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %reshape_7 = tensor.reshape %8(%cst_4) : (tensor<100xf32>, tensor<2xi64>) -> tensor<1x100xf32>
    %15 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%reshape_7 : tensor<1x100xf32>) outs(%15 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %17 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %10 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%17 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.subf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %19 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%18, %12 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%19 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.mulf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %21 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%20, %16 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%21 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.divf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %23 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %14 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%23 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.addf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %25 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %26 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<1x1xf32>) outs(%25 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %27 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%24, %26 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%27 : tensor<?x100xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.maximumf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    return %28 : tensor<?x100xf32>
  }
  module @__symbol__ {
  }
}



```

经过内存分析，给每一个张量添加inplace属性：

其中inplace 属性为 true就说明张量的内存可以被置换。

```
#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, 0)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x224xf32> {bufferization.access = "read", func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}) -> tensor<?x100xf32> attributes {entrance} {
    %cst = arith.constant 1.00000501 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x100xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x100xf32>
    %c0 = arith.constant 0 : index
    %cst_4 = arith.constant dense<[1, 100]> : tensor<2xi64>
    %cst_5 = arith.constant dense<[]> : tensor<100xf32>
    %cst_6 = arith.constant dense<[]> : tensor<224x100xf32>
    %dim = tensor.dim {__inplace_operands_attr__ = ["true", "none"]} %arg0, %c0 : tensor<?x224xf32>
    %0 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %1 = linalg.fill {__inplace_operands_attr__ = ["none", "true"]} ins(%cst_0 : f32) outs(%0 : tensor<?x100xf32>) -> tensor<?x100xf32>
    %2 = linalg.matmul {__inplace_operands_attr__ = ["true", "true", "true"]} ins(%arg0, %cst_6 : tensor<?x224xf32>, tensor<224x100xf32>) outs(%1 : tensor<?x100xf32>) -> tensor<?x100xf32>
    %reshape = tensor.reshape %cst_5(%cst_4) {__inplace_operands_attr__ = ["true", "true"]} : (tensor<100xf32>, tensor<2xi64>) -> tensor<1x100xf32>
    %3 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%reshape : tensor<1x100xf32>) outs(%3 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true"]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %5 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%5 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true", "true"]} {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.addf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %7 = bufferization.alloc_tensor() : tensor<100xf32>
    %8 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%7 : tensor<100xf32>) attrs =  {__inplace_operands_attr__ = ["false"]} {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<100xf32>
    %9 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1x100xf32>) outs(%9 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true"]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %11 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<1x100xf32>) outs(%11 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true"]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %13 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1x100xf32>) outs(%13 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true"]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %reshape_7 = tensor.reshape %8(%cst_4) {__inplace_operands_attr__ = ["true", "true"]} : (tensor<100xf32>, tensor<2xi64>) -> tensor<1x100xf32>
    %15 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%reshape_7 : tensor<1x100xf32>) outs(%15 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true"]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %17 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %10 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%17 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true", "true"]} {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.subf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %19 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%18, %12 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%19 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true", "true"]} {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.mulf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %21 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%20, %16 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%21 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true", "true"]} {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.divf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %23 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %14 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%23 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true", "true"]} {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.addf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    %25 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %26 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<1x1xf32>) outs(%25 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true"]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x100xf32>
    %27 = bufferization.alloc_tensor(%dim) : tensor<?x100xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%24, %26 : tensor<?x100xf32>, tensor<?x100xf32>) outs(%27 : tensor<?x100xf32>) attrs =  {__inplace_operands_attr__ = ["true", "true", "true"]} {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %29 = arith.maximumf %in, %in_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?x100xf32>
    return {__inplace_operands_attr__ = ["true"]} %28 : tensor<?x100xf32>
  }
  module @__symbol__ {
  }
}
```

根据inplace属性，将多余的张量分配去除并合并后：

可以看到整个运算当中分配内存的地方已经被清除了,在运行时不会分配任何内存。

```
// -----// IR Dump After InterpreterPass (transform-interpreter) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: memref<?x224xf32> {bufferization.access = "read", func.input_symbol_0 = "s0", func.input_symbol_1 = "c224"}, %arg1: memref<?x100xf32> {bufferize.result}) attributes {entrance} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.00000501 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %1 = memref.get_global @__constant_100xf32 : memref<1x100xf32>
    %2 = memref.get_global @__constant_224x100xf32 : memref<224x100xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x224xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %2 : memref<?x224xf32>, memref<224x100xf32>) outs(%arg1 : memref<?x100xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %3 = arith.mulf %in, %in_3 : f32
      %4 = arith.addf %cst, %3 : f32
      linalg.yield %4 : f32
    }
    linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %1: memref<?x100xf32>, memref<1x100xf32>) outs(%arg1 : memref<?x100xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %3 = arith.addf %in, %in_3 : f32
      %4 = arith.divf %3, %cst_0 : f32
      %5 = arith.addf %4, %cst : f32
      %6 = arith.maximumf %5, %cst : f32
      linalg.yield %6 : f32
    }
    return
  }
  module @__symbol__ {
  }
}

```
