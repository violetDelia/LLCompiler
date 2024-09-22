# Op合法化处理

    现在终于将神经网络从torch框架接入到MLIR系统了，可以开始正式的编译优化阶段了，但是由于为了和框架对接，定义了太多意义相似的Op以及类似WeightOp的特殊Op，需要将他们处理成方便优化的形式，这样之后的优化开发会简单很多。

    下图是alexnet转换的IR图，alexnet的结构已经算是十分简单了，但是在现在IR上查看查看它却十分的繁琐。十分不利于优化pass的实现。

```
#map = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>
#map1 = affine_map<()[s0, s1] -> (s0, 64, (s1 - 7) floordiv 4 + 1, (s1 - 7) floordiv 4 + 1)>
#map2 = affine_map<()[s0, s1] -> (s0, 64, ((s1 - 7) floordiv 4 - 1) floordiv 2 + 1, ((s1 - 7) floordiv 4 - 1) floordiv 2 + 1)>
#map3 = affine_map<()[s0, s1] -> (s0, 192, ((s1 - 7) floordiv 4 - 1) floordiv 2 + 1, ((s1 - 7) floordiv 4 - 1) floordiv 2 + 1)>
#map4 = affine_map<()[s0, s1] -> (s0, 192, (((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2, (((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2)>
#map5 = affine_map<()[s0, s1] -> (s0, 384, (((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2, (((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2)>
#map6 = affine_map<()[s0, s1] -> (s0, 256, (((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2, (((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2)>
#map7 = affine_map<()[s0, s1] -> (s0, 256, ((((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2 - 3) floordiv 2 + 1, ((((s1 - 7) floordiv 4 - 1) floordiv 2) floordiv 2 - 3) floordiv 2 + 1)>
#map8 = affine_map<()[s0, s1] -> (s0, 256, 6, 6)>
#map9 = affine_map<()[s0, s1] -> (s0, 9216)>
#map10 = affine_map<()[s0, s1] -> (s0, 4096)>
#map11 = affine_map<()[s0, s1] -> (s0, 1000)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: i64, %arg1: i64, %arg2: tensor<?x3x?x?xf32>) -> tensor<?x1000xf32> {
    %0 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_0.weight.npy"}> : () -> tensor<64x3x11x11xf32>
    %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_0.bias.npy"}> : () -> tensor<64xf32>
    %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_3.weight.npy"}> : () -> tensor<192x64x5x5xf32>
    %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_3.bias.npy"}> : () -> tensor<192xf32>
    %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_6.weight.npy"}> : () -> tensor<384x192x3x3xf32>
    %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_6.bias.npy"}> : () -> tensor<384xf32>
    %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_8.weight.npy"}> : () -> tensor<256x384x3x3xf32>
    %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_8.bias.npy"}> : () -> tensor<256xf32>
    %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_10.weight.npy"}> : () -> tensor<256x256x3x3xf32>
    %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___features_10.bias.npy"}> : () -> tensor<256xf32>
    %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___classifier_1.weight.npy"}> : () -> tensor<4096x9216xf32>
    %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___classifier_1.bias.npy"}> : () -> tensor<4096xf32>
    %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___classifier_4.weight.npy"}> : () -> tensor<4096x4096xf32>
    %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___classifier_4.bias.npy"}> : () -> tensor<4096xf32>
    %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___classifier_6.weight.npy"}> : () -> tensor<1000x4096xf32>
    %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-22T15:01:41.404546+08:00/L__self___classifier_6.bias.npy"}> : () -> tensor<1000xf32>
    %16 = "llh.symbolic_int"() <{value = "s0"}> : () -> i64
    %17 = "llh.symbolic_int"() <{value = "s2"}> : () -> i64
    "llh.symbolic_bind"(%arg2, %16, %17) <{expressions = #map}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %18 = "llh.conv_bias"(%arg2, %0, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 11, 11>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 4, 4>}> : (tensor<?x3x?x?xf32>, tensor<64x3x11x11xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%18, %16, %17) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %19 = "llh.relu"(%18) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%19, %16, %17) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %20 = "llh.max_pool"(%19) <{ceil_mode = true, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%20, %16, %17) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %21 = "llh.conv_bias"(%20, %2, %3) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<192x64x5x5xf32>, tensor<192xf32>) -> tensor<?x192x?x?xf32>
    "llh.symbolic_bind"(%21, %16, %17) <{expressions = #map3}> : (tensor<?x192x?x?xf32>, i64, i64) -> ()
    %22 = "llh.relu"(%21) : (tensor<?x192x?x?xf32>) -> tensor<?x192x?x?xf32>
    "llh.symbolic_bind"(%22, %16, %17) <{expressions = #map3}> : (tensor<?x192x?x?xf32>, i64, i64) -> ()
    %23 = "llh.max_pool"(%22) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x192x?x?xf32>) -> tensor<?x192x?x?xf32>
    "llh.symbolic_bind"(%23, %16, %17) <{expressions = #map4}> : (tensor<?x192x?x?xf32>, i64, i64) -> ()
    %24 = "llh.conv_bias"(%23, %4, %5) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x192x?x?xf32>, tensor<384x192x3x3xf32>, tensor<384xf32>) -> tensor<?x384x?x?xf32>
    "llh.symbolic_bind"(%24, %16, %17) <{expressions = #map5}> : (tensor<?x384x?x?xf32>, i64, i64) -> ()
    %25 = "llh.relu"(%24) : (tensor<?x384x?x?xf32>) -> tensor<?x384x?x?xf32>
    "llh.symbolic_bind"(%25, %16, %17) <{expressions = #map5}> : (tensor<?x384x?x?xf32>, i64, i64) -> ()
    %26 = "llh.conv_bias"(%25, %6, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x384x?x?xf32>, tensor<256x384x3x3xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%26, %16, %17) <{expressions = #map6}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %27 = "llh.relu"(%26) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%27, %16, %17) <{expressions = #map6}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %28 = "llh.conv_bias"(%27, %8, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%28, %16, %17) <{expressions = #map6}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %29 = "llh.relu"(%28) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%29, %16, %17) <{expressions = #map6}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %30 = "llh.max_pool"(%29) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%30, %16, %17) <{expressions = #map7}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %31 = "llh.adaptive_average_pool"(%30) : (tensor<?x256x?x?xf32>) -> tensor<?x256x6x6xf32>
    "llh.symbolic_bind"(%31, %16) <{expressions = #map8}> : (tensor<?x256x6x6xf32>, i64) -> ()
    %32 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %33 = "llh.flatten"(%31, %32) : (tensor<?x256x6x6xf32>, i64) -> tensor<?x9216xf32>
    "llh.symbolic_bind"(%33, %16) <{expressions = #map9}> : (tensor<?x9216xf32>, i64) -> ()
    %34 = "llh.drop"(%33) <{p = 5.000000e-01 : f64}> : (tensor<?x9216xf32>) -> tensor<?x9216xf32>
    "llh.symbolic_bind"(%34, %16) <{expressions = #map9}> : (tensor<?x9216xf32>, i64) -> ()
    %35 = "llh.transpose"(%10) <{perms = array<i64: 1, 0>}> : (tensor<4096x9216xf32>) -> tensor<9216x4096xf32>
    %36 = "llh.matmul"(%34, %35) : (tensor<?x9216xf32>, tensor<9216x4096xf32>) -> tensor<?x4096xf32>
    "llh.symbolic_bind"(%36, %16) <{expressions = #map10}> : (tensor<?x4096xf32>, i64) -> ()
    %37 = "llh.relu"(%36) : (tensor<?x4096xf32>) -> tensor<?x4096xf32>
    "llh.symbolic_bind"(%37, %16) <{expressions = #map10}> : (tensor<?x4096xf32>, i64) -> ()
    %38 = "llh.drop"(%37) <{p = 5.000000e-01 : f64}> : (tensor<?x4096xf32>) -> tensor<?x4096xf32>
    "llh.symbolic_bind"(%38, %16) <{expressions = #map10}> : (tensor<?x4096xf32>, i64) -> ()
    %39 = "llh.transpose"(%12) <{perms = array<i64: 1, 0>}> : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %40 = "llh.matmul"(%38, %39) : (tensor<?x4096xf32>, tensor<4096x4096xf32>) -> tensor<?x4096xf32>
    "llh.symbolic_bind"(%40, %16) <{expressions = #map10}> : (tensor<?x4096xf32>, i64) -> ()
    %41 = "llh.relu"(%40) : (tensor<?x4096xf32>) -> tensor<?x4096xf32>
    "llh.symbolic_bind"(%41, %16) <{expressions = #map10}> : (tensor<?x4096xf32>, i64) -> ()
    %42 = "llh.transpose"(%14) <{perms = array<i64: 1, 0>}> : (tensor<1000x4096xf32>) -> tensor<4096x1000xf32>
    %43 = "llh.matmul"(%41, %42) : (tensor<?x4096xf32>, tensor<4096x1000xf32>) -> tensor<?x1000xf32>
    "llh.symbolic_bind"(%43, %16) <{expressions = #map11}> : (tensor<?x1000xf32>, i64) -> ()
    return %43 : tensor<?x1000xf32>
  }
}
```

## WeightOp

    此前为了加快构建xDSL，将模型的权重保存为临时文件，用WeightOp表达，但是WeightOp没有数据信息，现在需要将其转为ConstOp，以便在后续的pass中进行优化，比如常量折叠和数据切分，以及一些比较特殊的优化，比如将小通道的matmul、conv这些无法充分利用带宽的计算进行数据重排，来提高硬件资源的利用率。

    处理之后的IR如下所示。

```
%0 = "llh.constant"() <{value = dense<"XXXXXX"> : tensor<10x3x5x5xf32>}> : () -> tensor<10x3x5x5xf32>
    %1 = "llh.constant"() <{value = dense<[0.0967884659, -0.0216379575, -0.0143847112, -0.073352471, -0.0970872938, -0.015869746, 0.0640928224, 0.0531028621, -4.52734239E-4, 0.0977045372]> : tensor<10xf32>}> : () -> tensor<10xf32>
    %2 = "llh.constant"() <{value = dense<"XXXXXX"> : tensor<3x10x5x5xf32>}> : () -> tensor<3x10x5x5xf32>
    %3 = "llh.constant"() <{value = dense<[0.0222571176, 0.0185563602, -0.0255357195]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %4 = "llh.constant"() <{value = dense<"XXXXXX"> : tensor<2x110xf32>}> : () -> tensor<2x110xf32>
    %5 = "llh.constant"() <{value = dense<[-0.0888146609, -0.0846343562]> : tensor<2xf32>}> : () -> tensor<2xf32>
```
