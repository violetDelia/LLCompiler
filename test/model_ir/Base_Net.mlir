#map = affine_map<()[s0, s1] -> (s0, s1, 224, 224)>
#map1 = affine_map<()[s0, s1] -> (1, s0 * s1, 224, 224)>
#map2 = affine_map<()[s0, s1] -> (2, (s0 * s1) floordiv 2, 224, 224)>
#map3 = affine_map<()[s0, s1] -> (2, 10, 104, 104)>
#map4 = affine_map<()[s0, s1] -> (2, 3, 110, 110)>
#map5 = affine_map<()[s0, s1] -> (2, 3, 110, 2)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x?x224x224xf32>, %arg1: i64, %arg2: i64) -> tensor<2x3x110x2xf32> {
    %0 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-27T23:20:33.415740+08:00/L__self___conv_layer1.weight.npy"}> : () -> tensor<10x3x5x5xf32>
    %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-27T23:20:33.415740+08:00/L__self___conv_layer1.bias.npy"}> : () -> tensor<10xf32>
    %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-27T23:20:33.415740+08:00/L__self___conv_layer2.weight.npy"}> : () -> tensor<3x10x5x5xf32>
    %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-27T23:20:33.415740+08:00/L__self___conv_layer2.bias.npy"}> : () -> tensor<3xf32>
    %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-27T23:20:33.415740+08:00/L__self___cf.weight.npy"}> : () -> tensor<2x110xf32>
    %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-27T23:20:33.415740+08:00/L__self___cf.bias.npy"}> : () -> tensor<2xf32>
    %6 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
    %7 = "llh.torch_symbolic_int"() <{sym_name = "s1"}> : () -> i64
    "llh.symbolic_bind"(%arg0, %6, %7) <{expressions = #map}> : (tensor<?x?x224x224xf32>, i64, i64) -> ()
    %8 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %9 = "llh.dim"(%arg0, %8) : (tensor<?x?x224x224xf32>, i64) -> i64
    %10 = "llh.constant"() <{value = 3 : i64}> : () -> i64
    %11 = "llh.dim"(%arg0, %10) : (tensor<?x?x224x224xf32>, i64) -> i64
    %12 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %13 = "llh.constant"() <{value = 6 : i64}> : () -> i64
    %14 = "llh.reshape"(%arg0, %12, %13, %9, %11) : (tensor<?x?x224x224xf32>, i64, i64, i64, i64) -> tensor<1x?x224x224xf32>
    "llh.symbolic_bind"(%14, %7, %6) <{expressions = #map1}> : (tensor<1x?x224x224xf32>, i64, i64) -> ()
    %15 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %16 = "llh.constant"() <{value = 3 : i64}> : () -> i64
    %17 = "llh.constant"() <{value = 224 : i64}> : () -> i64
    %18 = "llh.constant"() <{value = 224 : i64}> : () -> i64
    %19 = "llh.reshape"(%14, %15, %16, %17, %18) : (tensor<1x?x224x224xf32>, i64, i64, i64, i64) -> tensor<2x?x224x224xf32>
    "llh.symbolic_bind"(%19, %7, %6) <{expressions = #map2}> : (tensor<2x?x224x224xf32>, i64, i64) -> ()
    %20 = "llh.conv_bias"(%19, %0, %1) <{dilation = array<i64: 5, 5>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>}> : (tensor<2x?x224x224xf32>, tensor<10x3x5x5xf32>, tensor<10xf32>) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%20) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %21 = "llh.add"(%20, %20) : (tensor<2x10x104x104xf32>, tensor<2x10x104x104xf32>) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%21) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %22 = "llh.constant"() <{value = 5.33333302 : f32}> : () -> f32
    %23 = "llh.div"(%20, %22) : (tensor<2x10x104x104xf32>, f32) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%23) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %24 = "llh.add"(%23, %21) : (tensor<2x10x104x104xf32>, tensor<2x10x104x104xf32>) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%24) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %25 = "llh.mul"(%23, %23) : (tensor<2x10x104x104xf32>, tensor<2x10x104x104xf32>) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%25) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %26 = "llh.add"(%24, %25) : (tensor<2x10x104x104xf32>, tensor<2x10x104x104xf32>) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%26) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %27 = "llh.add"(%26, %21) : (tensor<2x10x104x104xf32>, tensor<2x10x104x104xf32>) -> tensor<2x10x104x104xf32>
    "llh.symbolic_bind"(%27) <{expressions = #map3}> : (tensor<2x10x104x104xf32>) -> ()
    %28 = "llh.conv_bias"(%27, %2, %3) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, pad = array<i64: 5, 5, 5, 5>, stride = array<i64: 1, 1>}> : (tensor<2x10x104x104xf32>, tensor<3x10x5x5xf32>, tensor<3xf32>) -> tensor<2x3x110x110xf32>
    "llh.symbolic_bind"(%28) <{expressions = #map4}> : (tensor<2x3x110x110xf32>) -> ()
    %29 = "llh.mul"(%28, %28) : (tensor<2x3x110x110xf32>, tensor<2x3x110x110xf32>) -> tensor<2x3x110x110xf32>
    "llh.symbolic_bind"(%29) <{expressions = #map4}> : (tensor<2x3x110x110xf32>) -> ()
    %30 = "llh.add"(%28, %29) : (tensor<2x3x110x110xf32>, tensor<2x3x110x110xf32>) -> tensor<2x3x110x110xf32>
    "llh.symbolic_bind"(%30) <{expressions = #map4}> : (tensor<2x3x110x110xf32>) -> ()
    %31 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %32 = "llh.div"(%28, %31) : (tensor<2x3x110x110xf32>, i64) -> tensor<2x3x110x110xf32>
    "llh.symbolic_bind"(%32) <{expressions = #map4}> : (tensor<2x3x110x110xf32>) -> ()
    %33 = "llh.add"(%30, %32) : (tensor<2x3x110x110xf32>, tensor<2x3x110x110xf32>) -> tensor<2x3x110x110xf32>
    "llh.symbolic_bind"(%33) <{expressions = #map4}> : (tensor<2x3x110x110xf32>) -> ()
    %34 = "llh.transpose"(%4) <{perms = array<i64: 1, 0>}> : (tensor<2x110xf32>) -> tensor<110x2xf32>
    %35 = "llh.matmul"(%33, %34) : (tensor<2x3x110x110xf32>, tensor<110x2xf32>) -> tensor<2x3x110x2xf32>
    %36 = "llh.add"(%35, %5) : (tensor<2x3x110x2xf32>, tensor<2xf32>) -> tensor<2x3x110x2xf32>
    "llh.symbolic_bind"(%36) <{expressions = #map5}> : (tensor<2x3x110x2xf32>) -> ()
    return %36 : tensor<2x3x110x2xf32>
  }
}
