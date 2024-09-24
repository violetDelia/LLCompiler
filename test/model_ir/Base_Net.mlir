// -----// IR Dump Before Operationlegalization (operation-legalization) //----- //
#map = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>
#map1 = affine_map<()[s0, s1] -> (s0, 10, (s1 - 17) floordiv 2 + 1, (s1 - 17) floordiv 2 + 1)>
#map2 = affine_map<()[s0, s1] -> (s0, 3, (s1 - 17) floordiv 2 + 7, (s1 - 17) floordiv 2 + 7)>
#map3 = affine_map<()[s0, s1] -> (s0, 3, (s1 - 17) floordiv 2 + 7, 2)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: i64, %arg1: i64, %arg2: tensor<?x3x?x?xf32>) -> tensor<?x3x?x2xf32> {
    %0 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T02:50:04.753774+08:00/L__self___conv_layer1.weight.npy"}> : () -> tensor<10x3x5x5xf32>
    %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T02:50:04.753774+08:00/L__self___conv_layer1.bias.npy"}> : () -> tensor<10xf32>
    %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T02:50:04.753774+08:00/L__self___conv_layer2.weight.npy"}> : () -> tensor<3x10x5x5xf32>
    %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T02:50:04.753774+08:00/L__self___conv_layer2.bias.npy"}> : () -> tensor<3xf32>
    %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T02:50:04.753774+08:00/L__self___cf.weight.npy"}> : () -> tensor<2x110xf32>
    %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T02:50:04.753774+08:00/L__self___cf.bias.npy"}> : () -> tensor<2xf32>
    %6 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
    %7 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
    "llh.symbolic_bind"(%arg2, %6, %7) <{expressions = #map}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %8 = "llh.conv_bias"(%arg2, %0, %1) <{dilation = array<i64: 5, 5>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<10x3x5x5xf32>, tensor<10xf32>) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%8, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %9 = "llh.add"(%8, %8) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%9, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %10 = "llh.constant"() <{value = 5.33333302 : f32}> : () -> f32
    %11 = "llh.div"(%8, %10) : (tensor<?x10x?x?xf32>, f32) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%11, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %12 = "llh.add"(%11, %9) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%12, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %13 = "llh.mul"(%11, %11) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%13, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %14 = "llh.add"(%12, %13) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%14, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %15 = "llh.add"(%14, %9) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
    "llh.symbolic_bind"(%15, %6, %7) <{expressions = #map1}> : (tensor<?x10x?x?xf32>, i64, i64) -> ()
    %16 = "llh.conv_bias"(%15, %2, %3) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, pad = array<i64: 5, 5, 5, 5>, stride = array<i64: 1, 1>}> : (tensor<?x10x?x?xf32>, tensor<3x10x5x5xf32>, tensor<3xf32>) -> tensor<?x3x?x?xf32>
    "llh.symbolic_bind"(%16, %6, %7) <{expressions = #map2}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %17 = "llh.mul"(%16, %16) : (tensor<?x3x?x?xf32>, tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
    "llh.symbolic_bind"(%17, %6, %7) <{expressions = #map2}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %18 = "llh.add"(%16, %17) : (tensor<?x3x?x?xf32>, tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
    "llh.symbolic_bind"(%18, %6, %7) <{expressions = #map2}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %19 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %20 = "llh.div"(%16, %19) : (tensor<?x3x?x?xf32>, i64) -> tensor<?x3x?x?xf32>
    "llh.symbolic_bind"(%20, %6, %7) <{expressions = #map2}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %21 = "llh.add"(%18, %20) : (tensor<?x3x?x?xf32>, tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
    "llh.symbolic_bind"(%21, %6, %7) <{expressions = #map2}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %22 = "llh.transpose"(%4) <{perms = array<i64: 1, 0>}> : (tensor<2x110xf32>) -> tensor<110x2xf32>
    %23 = "llh.matmul"(%21, %22) : (tensor<?x3x?x?xf32>, tensor<110x2xf32>) -> tensor<?x3x?x2xf32>
    %24 = "llh.add"(%23, %5) : (tensor<?x3x?x2xf32>, tensor<2xf32>) -> tensor<?x3x?x2xf32>
    "llh.symbolic_bind"(%24, %6, %7) <{expressions = #map3}> : (tensor<?x3x?x2xf32>, i64, i64) -> ()
    return %24 : tensor<?x3x?x2xf32>
  }
}


