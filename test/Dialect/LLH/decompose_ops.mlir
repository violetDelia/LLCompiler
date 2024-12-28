// RUN: llc-opt --split-input-file --decompose-ops --canonicalize --cse %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --decompose-ops --cse /home/lfr/LLCompiler/test/Dialect/LLH/decompose_ops.mlir

func.func @batch_norm(%arg0: tensor<?x3x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}) -> tensor<?x3x?x?xf32> attributes {entrance} {
    %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<3xf32>
    %1 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<3xf32>
    %2 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<3xf32>
    %3 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<3xf32>
    %4 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<1xi64>
    // CHECK: llh.add
    // CHECK: llh.sqrt
    // CHECK: llh.reshape
    // CHECK: llh.broadcast_to
    // CHECK: llh.reshape
    // CHECK: llh.broadcast_to
    // CHECK: llh.sub
    // CHECK: llh.mul
    // CHECK: llh.div
    // CHECK: llh.add
    // CHECK-NOT: llh.batch_norm_inference
    %5 = "llh.batch_norm_inference"(%arg0, %0, %1, %2, %3) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<?x3x?x?xf32>
    return %5 : tensor<?x3x?x?xf32>
  }


