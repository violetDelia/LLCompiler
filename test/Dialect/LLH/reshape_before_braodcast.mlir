// RUN: llc-opt --split-input-file --reshape-before-braodcast %s| FileCheck %s


module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @main(%arg0: tensor<?x?x224x224xf32>) -> tensor<?x224x?x224xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 224 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = 0 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T12:16:58.987739+08:00/L__self___cf.weight.npy"}> : () -> tensor<1x224xf32>
    %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-02T12:16:58.987739+08:00/L__self___cf.bias.npy"}> : () -> tensor<1xf32>
    %6 = "llh.transpose"(%4) <{perms = array<i64: 1, 0>}> : (tensor<1x224xf32>) -> tensor<224x1xf32>
    %7 = "llh.matmul"(%arg0, %6) : (tensor<?x?x224x224xf32>, tensor<224x1xf32>) -> tensor<?x?x224x1xf32>
    %8 = "llh.add"(%7, %5) : (tensor<?x?x224x1xf32>, tensor<1xf32>) -> tensor<?x?x224x1xf32>
    %9 = "llh.add"(%arg0, %8) : (tensor<?x?x224x224xf32>, tensor<?x?x224x1xf32>) -> tensor<?x?x224x224xf32>
    %10 = "llh.add"(%9, %2) : (tensor<?x?x224x224xf32>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<?x?x224x224xf32>
    %11 = "llh.add"(%10, %2) : (tensor<?x?x224x224xf32>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<?x?x224x224xf32>
    %12 = "llh.mul"(%11, %11) : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
    %13 = "llh.dim"(%12, %1) : (tensor<?x?x224x224xf32>, i64) -> i64
    %14 = "llh.dim"(%12, %3) : (tensor<?x?x224x224xf32>, i64) -> i64
    %15 = "llh.reshape"(%12, %13, %0, %14, %0) : (tensor<?x?x224x224xf32>, i64, i64, i64, i64) -> tensor<?x224x?x224xf32>
    %16 = "llh.sub"(%15, %2) : (tensor<?x224x?x224xf32>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<?x224x?x224xf32>
    return %16 : tensor<?x224x?x224xf32>
  }
}
