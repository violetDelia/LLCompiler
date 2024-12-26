# 算子融合专题<1> elementwise fusion

# 引言

Ai编译器狭义上来讲可以认为是一个计算图的优化“引擎”，在计算图层面，主要的作用有两方面，一方面是在图层面进行等价图变换，来减少模型的计算量；另一方面是进行融合算子的“匹配”，减少算子的实际访存量。能够做到这两件事情就可以称它是一个"Ai编译器"了。当然普遍上，编译器会做适配后端硬件的代码生成，让硬件的资源充分利用起来。如果是纯JIT运行的AI编译器，算子融合的优化性能很依赖代码生成所采用的“策略”，TVM就是这样的，会在一系列代码生成的“策略”中搜索一个性能最高的算子，但是搜索的时间很长，尤其是比较复杂的算子。

一般算子可分为两类，访存密集型和计算密集型，主要是看算子的访存比。算子融合它对访存密集型的算子优化效果比较明显。原因就是事实上算子融合只是减少实际计算的访存量，实际上运行的指令数是不会变的。

对于单个算子计算来说，这个熟悉算子开发的应该了解更深刻，可以简单分为三种模式，一对一、一对多、和多对多。一对一就是elewise类的算子，一对多就是reduce类的算子，多对多就是conv，dot这类计算密集型的算子。

像Trition据我了解也只是做了elewise的融合优化。reduce类的融合和更复杂的的融合也没有做。（大概因为这样更有性价比吧）

这个专题不会涉及融合算子的代码生成优化，因为这些算是编译器其他优化手段，不能归为算子融合。就先从elewise 融合开始一点点介绍吧。

# 实例模型

这是一个简单的能够完美进行elewise fusion的一个demo

```
class ElewiseFusion1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rule = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x - 3
        x = x / 2
        x_max = self.rule(x)
        x = x_max * x
        return x
```

我们从前端获取到的计算图是这样的：

```
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @main(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>> attributes {entrance} {
    %0 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %3 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %5 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %6 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    %7 = "llh.reshape"(%2, %5, %5, %5, %5) : (tensor<1xf32, #llh.encoding<shapes = @c1>>, i64, i64, i64, i64) -> tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>
    %8 = "llh.dim"(%6, %4) <{symbol = @s0}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %9 = "llh.dim"(%6, %5) <{symbol = @s1}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %10 = "llh.dim"(%6, %1) <{symbol = @s2}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %11 = "llh.dim"(%6, %3) <{symbol = @s2}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %12 = "llh.broadcast_to"(%7, %8, %9, %10, %11) <{cast_dims = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>, i64, i64, i64, i64) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    %13 = "llh.sub"(%6, %12) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    %14 = "llh.reshape"(%0, %5, %5, %5, %5) : (tensor<1xf32, #llh.encoding<shapes = @c1>>, i64, i64, i64, i64) -> tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>
    %15 = "llh.dim"(%13, %4) <{symbol = @s0}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %16 = "llh.dim"(%13, %5) <{symbol = @s1}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %17 = "llh.dim"(%13, %1) <{symbol = @s2}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %18 = "llh.dim"(%13, %3) <{symbol = @s2}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, i64) -> i64
    %19 = "llh.broadcast_to"(%14, %15, %16, %17, %18) <{cast_dims = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>, i64, i64, i64, i64) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    %20 = "llh.div"(%13, %19) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    %21 = "llh.relu"(%20) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    %22 = "llh.mul"(%21, %20) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
    return %22 : tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s2>>
  }
  module @__symbol__ {
  }
}
```

如果是单算子去调用的话，它实际的数据流是这样的：

即：add --> sub --> div --> rule --> mul

在3级访存的结构下：

即 input(L3) --> L2 -->L1 --> add --> L2 -->out (L3) --> L2 --> L1 -->sub -->.......

每次计算都有将计算数据从L3 搬运到L1给寄存器计算，然后搬回L3上在进行下一个算子的操作。

但如果是融合算子的话：

即 fusion（mul、sub、div、rule、mul）

它的数据流是这样的：

input（L3）--> L2 --> L1 --> add --> sub --> ... -->L2 --> out(L3)

这样就节省了很多的Dma操作，这也是为什么融合算子计算快的原因。

## 融合结果

计算图下降到linalg上的表示是这样的：

```
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  func.func @main(%arg0: tensor<200x3x224x224xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c224"}) -> tensor<200x3x224x224xf32> attributes {entrance} {
    %cst = arith.constant dense<0.000000e+00> : tensor<200x3x224x224xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<200x3x224x224xf32>
    %cst_1 = arith.constant dense<3.000000e+00> : tensor<200x3x224x224xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<200x3x224x224xf32>, tensor<200x3x224x224xf32>) outs(%arg0 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %cst_1 : tensor<200x3x224x224xf32>, tensor<200x3x224x224xf32>) outs(%0 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.subf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %cst_0 : tensor<200x3x224x224xf32>, tensor<200x3x224x224xf32>) outs(%1 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.divf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    %3 = tensor.empty() : tensor<200x3x224x224xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst : tensor<200x3x224x224xf32>, tensor<200x3x224x224xf32>) outs(%3 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.maximumf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %2 : tensor<200x3x224x224xf32>, tensor<200x3x224x224xf32>) outs(%4 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    return %5 : tensor<200x3x224x224xf32>
  }
  module @__symbol__ {
  }
}
```

每个linalg.generic就可以代表一个小的算子，当他融合后会变成这样：

```
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  func.func @main(%arg0: tensor<200x3x224x224xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c224"}) -> tensor<200x3x224x224xf32> attributes {entrance} {
    %cst = arith.constant 3.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<200x3x224x224xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<200x3x224x224xf32>) outs(%0 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %in : f32
      %3 = arith.subf %2, %cst : f32
      %4 = arith.divf %3, %cst_0 : f32
      %5 = arith.maximumf %4, %cst_1 : f32
      %6 = arith.mulf %5, %4 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    return %1 : tensor<200x3x224x224xf32>
  }
  module @__symbol__ {
  }
}
```

ps： mlir 原生的 linalg fusion 动态的支持不是很好，看来后续要支持动态的话要自己写一套。只好展示的静态的图啦。

## 计算时间记录

记录一下实际的计算时间：

torch eager模式：0.041s

torch compiler ：0.007s

llcompiler没有融合的 运行时间：0.121s

llcompiler 融合后的计算时间     ：0.020s

这个是我简单记录的结果，但是看起来差不多，这个模型融合要比单算子计算快了6倍。现在我的编译器在elewise算子已经比torch 的eager 模式下还要快了。 而且后续还有unroll，向量化等优化手段还没有应用上。想达到最极致的性能，还需要不断的尝试找到循环展开次数/数据切分的大小。

嗯就暂时这样。element wise fusion 的数据流和计算流很简单，很适合入门。

这个我大概后续会更新reduce + 和 dot 的fusion 以及自动融合（其实跟TVM的思路没多大区别，当然我自己写肯定不会写的像他那么复杂，支配树也是很难写的好吧）的方法。

现在主流的做法大致是 conv / matmul + reduce + elewise 这样的方式去融合。但事实上好像elewise的提升幅度（或者叫做性价比吧）是最高的。如果涉及到reduce\dot\matmul的融合的话，是需要考虑L1的大小去做切分的，这个就很复杂了，因为elewise不需要关心这些东西，所以很好实现。

但除此之外还有比较定制化的fusion 模板，还有横向的算子fusion这个专题应该不会涉及到。
