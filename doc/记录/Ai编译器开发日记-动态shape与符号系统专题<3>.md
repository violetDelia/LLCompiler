# 符号折叠

&emsp;在专题二中提到符号优化的方案，后来发现在符号优化中由于关系众多难以发现对符号的关系推导过于困难，因此借助符号计算库symengine对其进行改进，同时也实现了符号优化的第一个功能，emm就称作符号折叠吧。它的用意是将相同符号的值进行化简，这样做的好处是方便后续的动态内存分析以及循环优化。

&emsp;symengine是MIT协议的的符号计算库，不仅可以进行表达符号之间的计算，还可以在符号比较中添加约束进行比较。只要构建出符号之间的计算关系，可以很方便的对符号进行推演和比较。akg-mlir中也用到了symengine，但是看起来只是用到了最简单的功能。
&emsp;具体用法详见官方文档和源码中的test文件夹。

## 改进后的方案

&emsp;之前在推导方案如果对复杂的op进行符号推导，比如conv、cat等会产生大量的关系，导致推演简化困难。于是借助symengine实现更高效的推导方案。
&emsp;以conv 和 slice为例：

```
func.func @conv(%arg0: tensor<?x3x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"} ) ->() attributes {entrance}{
        %4 = "llh.weight"() <{weight_file = "npy"}> : () -> tensor<64x3x7x7xf32>
        %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<*xf32>
    return 
    }
```

&emsp;进行推导后后：

```mlir
#map = affine_map<(d0)[s0] -> ((s0 - 1) ceildiv 2 + 1)>
module {
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @conv(%arg0: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}) attributes {entrance} {
    %0 = "llh.weight"() <{weight_file = "npy"}> : () -> tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>
    %1 = "llh.conv"(%arg0, %0) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s1>>
    return
  }
  module @__symbol__ {
    "llh.symbol_relation_map"() <{express = "1 + (1.0/2.0)*(-1 + s2)", relation = #map, relations = [@s2], symbol = @s1}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s1}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s2}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s0}> : () -> ()
  }
}
```

&emsp;在llh.symbol_relation_map中，符号s1就是conv计算之后的dim值, 它与输入s2的关系表达式为`1 + (1.0/2.0)*(-1 + s2)`, 同时也在map中记录了s2和s1的关系 `#map = affine_map<(d0)[s0] -> ((s0 - 1) ceildiv 2 + 1)>`。这是为了之后多面体优化而构造的值。
&emsp;`1 + (1.0/2.0)*(-1 + s2)`就是使用symengine所构造的表达式，对于相同的符号就没有必要添加新的符号了。

## 改进之后推导resnet

&emsp;在之前的推导方案中，推导resnet会生成16个不同的符号，这其中有很多符号的值是相等的，但是因为关系复杂且错乱，很难推演出哪几个符号是相同的，但是在新的方案中，只有不同的表达式才会生成新的符号，于是就没有必要进行复杂的推演了。
&emsp;推导前：

```mlir
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c256"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @main(%arg0: tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<1x10xf32> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c10, value = 10 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c512, value = 512 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c256, value = 256 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %5 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<64x3x7x7xf32>
    ......
    %99 = "llh.weight"() <{weight_file = ".npy"}> : () -> tensor<1xi64>
    %100 = "llh.conv"(%arg0, %6) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>>, tensor<64x3x7x7xf32>) -> tensor<1x64x?x?xf32>
    %101 = "llh.batch_norm"(%100, %7, %8, %55, %56) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %102 = "llh.relu"(%101) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %103 = "llh.max_pool"(%102) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %104 = "llh.conv"(%103, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %105 = "llh.batch_norm"(%104, %10, %11, %58, %59) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %106 = "llh.relu"(%105) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %107 = "llh.conv"(%106, %12) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %108 = "llh.batch_norm"(%107, %13, %14, %61, %62) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %109 = "llh.add"(%108, %103) : (tensor<1x64x?x?xf32>, tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %110 = "llh.relu"(%109) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %111 = "llh.conv"(%110, %15) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %112 = "llh.batch_norm"(%111, %16, %17, %64, %65) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %113 = "llh.relu"(%112) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %114 = "llh.conv"(%113, %18) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %115 = "llh.batch_norm"(%114, %19, %20, %67, %68) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %116 = "llh.add"(%115, %110) : (tensor<1x64x?x?xf32>, tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %117 = "llh.relu"(%116) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %118 = "llh.conv"(%117, %21) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x?x?xf32>
    %119 = "llh.batch_norm"(%118, %22, %23, %70, %71) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %120 = "llh.relu"(%119) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %121 = "llh.conv"(%120, %24) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x?x?xf32>
    %122 = "llh.batch_norm"(%121, %25, %26, %73, %74) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %123 = "llh.conv"(%117, %27) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<1x128x?x?xf32>
    %124 = "llh.batch_norm"(%123, %28, %29, %76, %77) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %125 = "llh.add"(%122, %124) : (tensor<1x128x?x?xf32>, tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %126 = "llh.relu"(%125) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %127 = "llh.conv"(%126, %30) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x?x?xf32>
    %128 = "llh.batch_norm"(%127, %31, %32, %79, %80) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %129 = "llh.relu"(%128) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %130 = "llh.conv"(%129, %33) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x?x?xf32>
    %131 = "llh.batch_norm"(%130, %34, %35, %82, %83) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %132 = "llh.add"(%131, %126) : (tensor<1x128x?x?xf32>, tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %133 = "llh.relu"(%132) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %134 = "llh.conv"(%133, %36) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<1x256x?x?xf32>
    %135 = "llh.batch_norm"(%134, %37, %38, %85, %86) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x?x?xf32>
    %136 = "llh.relu"(%135) : (tensor<1x256x?x?xf32>) -> tensor<1x256x?x?xf32>
    %137 = "llh.conv"(%136, %39) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x?x?xf32>
    %138 = "llh.batch_norm"(%137, %40, %41, %88, %89) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x?x?xf32>
    %139 = "llh.conv"(%133, %42) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<1x256x?x?xf32>
    %140 = "llh.batch_norm"(%139, %43, %44, %91, %92) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x?x?xf32>
    %141 = "llh.add"(%138, %140) : (tensor<1x256x?x?xf32>, tensor<1x256x?x?xf32>) -> tensor<1x256x?x?xf32>
    %142 = "llh.relu"(%141) : (tensor<1x256x?x?xf32>) -> tensor<1x256x?x?xf32>
    %143 = "llh.conv"(%142, %45) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x?x?xf32>
    %144 = "llh.batch_norm"(%143, %46, %47, %94, %95) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x?x?xf32>
    %145 = "llh.relu"(%144) : (tensor<1x256x?x?xf32>) -> tensor<1x256x?x?xf32>
    %146 = "llh.conv"(%145, %48) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x?x?xf32>
    %147 = "llh.batch_norm"(%146, %49, %50, %97, %98) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x?x?xf32>
    %148 = "llh.add"(%147, %142) : (tensor<1x256x?x?xf32>, tensor<1x256x?x?xf32>) -> tensor<1x256x?x?xf32>
    %149 = "llh.relu"(%148) : (tensor<1x256x?x?xf32>) -> tensor<1x256x?x?xf32>
    %150 = "llh.dim"(%149, %4) : (tensor<1x256x?x?xf32>, i64) -> i64
    %151 = "llh.dim"(%149, %3) : (tensor<1x256x?x?xf32>, i64) -> i64
    %152 = "llh.mul"(%2, %150) : (i64, i64) -> i64
    %153 = "llh.mul"(%152, %151) : (i64, i64) -> i64
    %154 = "llh.reshape"(%149, %5, %153) : (tensor<1x256x?x?xf32>, i64, i64) -> tensor<1x?xf32>
    %155 = "llh.transpose"(%51) <{perms = array<i64: 1, 0>}> : (tensor<512x4096xf32>) -> tensor<4096x512xf32>
    %156 = "llh.matmul"(%154, %155) : (tensor<1x?xf32>, tensor<4096x512xf32>) -> tensor<1x512xf32>
    %157 = "llh.reshape"(%52, %5, %1) : (tensor<512xf32>, i64, i64) -> tensor<1x512xf32>
    %158 = "llh.add"(%156, %157) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %159 = "llh.reshape"(%158, %5, %1) : (tensor<1x512xf32>, i64, i64) -> tensor<1x512xf32>
    %160 = "llh.transpose"(%53) <{perms = array<i64: 1, 0>}> : (tensor<10x512xf32>) -> tensor<512x10xf32>
    %161 = "llh.matmul"(%159, %160) : (tensor<1x512xf32>, tensor<512x10xf32>) -> tensor<1x10xf32>
    %162 = "llh.reshape"(%54, %5, %0) : (tensor<10xf32>, i64, i64) -> tensor<1x10xf32>
    %163 = "llh.add"(%161, %162) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %163 : tensor<1x10xf32>
  }
  module @__symbol__ {
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s1}> : () -> ()
  }
}
```

    推导之后：

```mlir
#map = affine_map<(d0)[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> ((s0 - 1) ceildiv 2 + 1)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "s6"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s5"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s4"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @main(%arg0: tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c10, value = 10 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c512, value = 512 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c256, value = 256 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %5 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    ''''''
    %100 = "llh.conv"(%arg0, %6) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s0>>
    %101 = "llh.batch_norm"(%100, %7, %8, %55, %56) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s0>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s0>>
    %102 = "llh.relu"(%101) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s0>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s0>>
    %103 = "llh.max_pool"(%102) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s0>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %104 = "llh.conv"(%103, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %105 = "llh.batch_norm"(%104, %10, %11, %58, %59) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %106 = "llh.relu"(%105) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %107 = "llh.conv"(%106, %12) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %108 = "llh.batch_norm"(%107, %13, %14, %61, %62) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %109 = "llh.add"(%108, %103) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %110 = "llh.relu"(%109) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %111 = "llh.conv"(%110, %15) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %112 = "llh.batch_norm"(%111, %16, %17, %64, %65) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %113 = "llh.relu"(%112) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %114 = "llh.conv"(%113, %18) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %115 = "llh.batch_norm"(%114, %19, %20, %67, %68) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %116 = "llh.add"(%115, %110) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %117 = "llh.relu"(%116) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>
    %118 = "llh.conv"(%117, %21) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<128x64x3x3xf32, #llh.encoding<shapes = @c128, @c64, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %119 = "llh.batch_norm"(%118, %22, %23, %70, %71) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %120 = "llh.relu"(%119) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %121 = "llh.conv"(%120, %24) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %122 = "llh.batch_norm"(%121, %25, %26, %73, %74) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %123 = "llh.conv"(%117, %27) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s2, @s2>>, tensor<128x64x1x1xf32, #llh.encoding<shapes = @c128, @c64, @c1, @c1>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %124 = "llh.batch_norm"(%123, %28, %29, %76, %77) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %125 = "llh.add"(%122, %124) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %126 = "llh.relu"(%125) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %127 = "llh.conv"(%126, %30) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %128 = "llh.batch_norm"(%127, %31, %32, %79, %80) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %129 = "llh.relu"(%128) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %130 = "llh.conv"(%129, %33) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %131 = "llh.batch_norm"(%130, %34, %35, %82, %83) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %132 = "llh.add"(%131, %126) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %133 = "llh.relu"(%132) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>
    %134 = "llh.conv"(%133, %36) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<256x128x3x3xf32, #llh.encoding<shapes = @c256, @c128, @c3, @c3>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %135 = "llh.batch_norm"(%134, %37, %38, %85, %86) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %136 = "llh.relu"(%135) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %137 = "llh.conv"(%136, %39) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256x256x3x3xf32, #llh.encoding<shapes = @c256, @c256, @c3, @c3>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %138 = "llh.batch_norm"(%137, %40, %41, %88, %89) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %139 = "llh.conv"(%133, %42) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s3, @s3>>, tensor<256x128x1x1xf32, #llh.encoding<shapes = @c256, @c128, @c1, @c1>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %140 = "llh.batch_norm"(%139, %43, %44, %91, %92) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %141 = "llh.add"(%138, %140) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %142 = "llh.relu"(%141) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %143 = "llh.conv"(%142, %45) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256x256x3x3xf32, #llh.encoding<shapes = @c256, @c256, @c3, @c3>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %144 = "llh.batch_norm"(%143, %46, %47, %94, %95) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %145 = "llh.relu"(%144) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %146 = "llh.conv"(%145, %48) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256x256x3x3xf32, #llh.encoding<shapes = @c256, @c256, @c3, @c3>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %147 = "llh.batch_norm"(%146, %49, %50, %97, %98) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %148 = "llh.add"(%147, %142) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %149 = "llh.relu"(%148) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>) -> tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>
    %150 = "llh.dim"(%149, %4) <{symbol = @s4}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, i64) -> i64
    %151 = "llh.dim"(%149, %3) <{symbol = @s4}> : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, i64) -> i64
    %152 = "llh.mul"(%2, %150) <{symbol = @s5}> : (i64, i64) -> i64
    %153 = "llh.mul"(%152, %151) <{symbol = @s6}> : (i64, i64) -> i64
    %154 = "llh.reshape"(%149, %5, %153) : (tensor<1x256x?x?xf32, #llh.encoding<shapes = @c1, @c256, @s4, @s4>>, i64, i64) -> tensor<1x?xf32, #llh.encoding<shapes = @c1, @s6>>
    %155 = "llh.transpose"(%51) <{perms = array<i64: 1, 0>}> : (tensor<512x4096xf32, #llh.encoding<shapes = @c512, @c4096>>) -> tensor<4096x512xf32, #llh.encoding<shapes = @c4096, @c512>>
    %156 = "llh.matmul"(%154, %155) : (tensor<1x?xf32, #llh.encoding<shapes = @c1, @s6>>, tensor<4096x512xf32, #llh.encoding<shapes = @c4096, @c512>>) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %157 = "llh.reshape"(%52, %5, %1) : (tensor<512xf32, #llh.encoding<shapes = @c512>>, i64, i64) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %158 = "llh.add"(%156, %157) : (tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>, tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %159 = "llh.reshape"(%158, %5, %1) : (tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>, i64, i64) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %160 = "llh.transpose"(%53) <{perms = array<i64: 1, 0>}> : (tensor<10x512xf32, #llh.encoding<shapes = @c10, @c512>>) -> tensor<512x10xf32, #llh.encoding<shapes = @c512, @c10>>
    %161 = "llh.matmul"(%159, %160) : (tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>, tensor<512x10xf32, #llh.encoding<shapes = @c512, @c10>>) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    %162 = "llh.reshape"(%54, %5, %0) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    %163 = "llh.add"(%161, %162) : (tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>, tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    return %163 : tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
  }
  module @__symbol__ {
    "llh.symbol_relation"() <{relation = @c4096, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s6}> : () -> ()
    "llh.symbol_relation_map"() <{express = "256*pow(1 + (1.0/4.0)*(-1 + (1.0/2.0)*(2 + (1.0/2.0)*(-1 + s1))), 2)", relation = #map, relations = [@s5, @s4], symbol = @s6}> : () -> ()
    "llh.symbol_relation_map"() <{express = "256*(1 + (1.0/4.0)*(-1 + (1.0/2.0)*(2 + (1.0/2.0)*(-1 + s1))))", relation = #map, relations = [@c256, @s4], symbol = @s5}> : () -> ()
    "llh.symbol_relation_map"() <{express = "1 + (1.0/4.0)*(-1 + (1.0/2.0)*(2 + (1.0/2.0)*(-1 + s1)))", relation = #map1, relations = [@s3], symbol = @s4}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s4}> : () -> ()
    "llh.symbol_relation_map"() <{express = "1 + (1.0/2.0)*(-1 + (1.0/2.0)*(2 + (1.0/2.0)*(-1 + s1)))", relation = #map1, relations = [@s2], symbol = @s3}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s3}> : () -> ()
    "llh.symbol_relation_map"() <{express = "(1.0/2.0)*(2 + (1.0/2.0)*(-1 + s1))", relation = #map1, relations = [@s0], symbol = @s2}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s2}> : () -> ()
    "llh.symbol_relation_map"() <{express = "1 + (1.0/2.0)*(-1 + s1)", relation = #map1, relations = [@s1], symbol = @s0}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s0}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s1}> : () -> ()
  }
}
```

&emsp;推导完后发现，其实整个网络只有6个不同的dim size，这样的信息对横向的算子融合提供了很方便的信息。因为横向的算子融合在动态图的情况下很难对算子的输入输出进行比较。dim值都是？而且很难找到输入输出的依赖关系。实际上很多的高效的算子融合在动态图的情况下是很难进行的。但是有了符号的信息后，可以轻松的知道他们的大小是不是一样的，这样两个尺寸一样的conv算子就可以合并成一个大的conv算子（一定情况下）。

## 符号折叠

&emsp;这个优化我称为是符号折叠，其实就是如果两个值的符号是相同的，那么就可以将他其中一个替换为另一个，这样不仅简化了IR图，也为后续的内存分析优化和循环优化提供基础和条件。
&emsp;以下模型为例：

```python
    class Extract(nn.Module):
        def__init__(self):
            super().__init__()

    def forward(self, x: torch.Tensor):
            x1 = x[1]
            x2 = x[-2]
            x3 = x1+x2
            x3 += x[0]
            return x3
```

&emsp;推导之后的计算图是这样的：

```mlir
    #map = affine_map<(d0)[s0, s1] -> (s0 + s1)>
    module attributes {builtin.gloabal_layout = "NCHW"} {
    "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
    "llh.symbolic_int"() <{sym_name = "c-2"}> : () -> ()
    "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
    "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
    "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
    "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
    "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
    func.func @main(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s1, @s1>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>> attributes {entrance} {
        %0 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
        %1 = "llh.constant"() <{symbol = @"c-2", value = -2 : i64}> : () -> i64
        %2 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
        %3 = "llh.extract"(%arg0, %2) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s1, @s1>>, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>
        %4 = "llh.dim"(%arg0, %0) <{symbol = @s0}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s1, @s1>>, i64) -> i64
        %5 = "llh.add"(%4, %1) <{symbol = @s2}> : (i64, i64) -> i64
        %6 = "llh.extract"(%arg0, %5) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s1, @s1>>, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>
        %7 = "llh.add"(%3, %6) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>
        %8 = "llh.extract"(%arg0, %0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s1, @s1>>, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>
        %9 = "llh.add"(%7, %8) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>
        return %9 : tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s1>>
    }
    module @__symbol__ {
        "llh.symbol_relation_map"() <{express = "-2 + s0", relation = #map, relations = [@s0, @"c-2"], symbol = @s2}> : () -> ()
        "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation `<GE>`, symbol = @s1}> : () -> ()
        "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation `<GE>`, symbol = @s0}> : () -> ()
    }
    }
```

在符号折叠之前是这样的：

```mlir
    #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #map1 = affine_map<(d0)[s0, s1] -> (s0 + s1)>
    module attributes {builtin.gloabal_layout = "NCHW"} {
    func.func @main(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<?x?x?xf32> attributes {entrance} {
        %c1 = arith.constant {symbol = @c1} 1 : index
        %c2 = arith.constant {symbol = @c2} 2 : index
        %c3 = arith.constant {symbol = @c3} 3 : index
        %c0 = arith.constant {symbol = @c0} 0 : index
        %c-2 = arith.constant {symbol = @"c-2"} -2 : index
        "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @s0, @s0, @s1, @s1>}> : (tensor<?x?x?x?xf32>) -> ()
        %dim = tensor.dim {symbol = @s0} %arg0, %c1 : tensor<?x?x?x?xf32>
        %dim_0 = tensor.dim {symbol = @s1} %arg0, %c2 : tensor<?x?x?x?xf32>
        %dim_1 = tensor.dim {symbol = @s1} %arg0, %c3 : tensor<?x?x?x?xf32>
        %extracted_slice = tensor.extract_slice %arg0[1, 0, 0, 0] [1, %dim, %dim_0, %dim_1] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x?x?xf32>
        %from_elements = tensor.from_elements %dim, %dim_0, %dim_1 : tensor<3xindex>
        %reshape = tensor.reshape %extracted_slice(%from_elements) : (tensor<1x?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
        %dim_2 = tensor.dim {symbol = @s0} %arg0, %c0 : tensor<?x?x?x?xf32>
        %0 = arith.addi %dim_2, %c-2 {symbol = @s2} : index
        %dim_3 = tensor.dim {symbol = @s0} %arg0, %c1 : tensor<?x?x?x?xf32>
        %dim_4 = tensor.dim {symbol = @s1} %arg0, %c2 : tensor<?x?x?x?xf32>
        %dim_5 = tensor.dim {symbol = @s1} %arg0, %c3 : tensor<?x?x?x?xf32>
        %extracted_slice_6 = tensor.extract_slice %arg0[%0, 0, 0, 0] [1, %dim_3, %dim_4, %dim_5] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x?x?xf32>
        %from_elements_7 = tensor.from_elements %dim_3, %dim_4, %dim_5 : tensor<3xindex>
        %reshape_8 = tensor.reshape %extracted_slice_6(%from_elements_7) : (tensor<1x?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
        %1 = tensor.empty(%dim, %dim_0, %dim_1) : tensor<?x?x?xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reshape, %reshape_8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%1 : tensor<?x?x?xf32>) {
        ^bb0(%in: f32, %in_15: f32, %out: f32):
        %5 = arith.addf %in, %in_15 : f32
        linalg.yield %5 : f32
        } -> tensor<?x?x?xf32>
        %dim_9 = tensor.dim {symbol = @s0} %arg0, %c1 : tensor<?x?x?x?xf32>
        %dim_10 = tensor.dim {symbol = @s1} %arg0, %c2 : tensor<?x?x?x?xf32>
        %dim_11 = tensor.dim {symbol = @s1} %arg0, %c3 : tensor<?x?x?x?xf32>
        %extracted_slice_12 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, %dim_9, %dim_10, %dim_11] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x?x?xf32>
        %from_elements_13 = tensor.from_elements %dim_9, %dim_10, %dim_11 : tensor<3xindex>
        %reshape_14 = tensor.reshape %extracted_slice_12(%from_elements_13) : (tensor<1x?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
        %3 = tensor.empty(%dim, %dim_0, %dim_1) : tensor<?x?x?xf32>
        %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %reshape_14 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) {
        ^bb0(%in: f32, %in_15: f32, %out: f32):
        %5 = arith.addf %in, %in_15 : f32
        linalg.yield %5 : f32
        } -> tensor<?x?x?xf32>
        return %4 : tensor<?x?x?xf32>
    }
    module @__symbol__ {
        "llh.symbol_relation_map"() <{express = "-2 + s0", relation = #map1, relations = [@s0, @"c-2"], symbol = @s2}> : () -> ()
        "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation `<GE>`, symbol = @s1}> : () -> ()
        "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation `<GE>`, symbol = @s0}> : () -> ()
    }
    }
```

&emsp;折叠之后是这样的：

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0)[s0, s1] -> (s0 + s1)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<?x?x?xf32> attributes {entrance} {
    %c1 = arith.constant {symbol = @c1} 1 : index
    %c2 = arith.constant {symbol = @c2} 2 : index
    %c-2 = arith.constant {symbol = @"c-2"} -2 : index
    "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @s0, @s0, @s1, @s1>}> : (tensor<?x?x?x?xf32>) -> ()
    %dim = tensor.dim {symbol = @s0} %arg0, %c1 : tensor<?x?x?x?xf32>
    %dim_0 = tensor.dim {symbol = @s1} %arg0, %c2 : tensor<?x?x?x?xf32>
    %extracted_slice = tensor.extract_slice %arg0[1, 0, 0, 0] [1, %dim, %dim_0, %dim_0] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x?x?xf32>
    %from_elements = tensor.from_elements %dim, %dim_0, %dim_0 : tensor<3xindex>
    %reshape = tensor.reshape %extracted_slice(%from_elements) : (tensor<1x?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
    %0 = arith.addi %dim, %c-2 {symbol = @s2} : index
    %extracted_slice_1 = tensor.extract_slice %arg0[%0, 0, 0, 0] [1, %dim, %dim_0, %dim_0] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x?x?xf32>
    %reshape_2 = tensor.reshape %extracted_slice_1(%from_elements) : (tensor<1x?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
    %1 = tensor.empty(%dim, %dim_0, %dim_0) : tensor<?x?x?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reshape, %reshape_2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%1 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %4 = arith.addf %in, %in_5 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, %dim, %dim_0, %dim_0] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x?x?xf32>
    %reshape_4 = tensor.reshape %extracted_slice_3(%from_elements) : (tensor<1x?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %reshape_4 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%1 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %4 = arith.addf %in, %in_5 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?xf32>
    return %3 : tensor<?x?x?xf32>
  }
  module @__symbol__ {
    "llh.symbol_relation_map"() <{express = "-2 + s0", relation = #map1, relations = [@s0, @"c-2"], symbol = @s2}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s1}> : () -> ()
    "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GE>, symbol = @s0}> : () -> ()
  }
}
```

&emsp;其中，`%dim_4 = tensor.dim {symbol = @s1} %arg0, %c2 : tensor<?x?x?x?xf32>` 和 `%dim_5 = tensor.dim {symbol = @s1} %arg0, %c3 : tensor<?x?x?x?xf32> `因为符号相同，所以被折叠为一个值了。这样为后续的内存分析和循环优化提供了更多的空间和遍历。虽然cse有类似的功能，但是在复杂的图中cse能够起到的作用就比较小了。
