// RUN: llc-opt --split-input-file --canonicalize --cse %s| FileCheck %s


module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  // CHECK-LABEL: simplyBinary
  func.func @simplyBinary(%arg0: tensor<?x?x?x?xf32>,%arg1: tensor<1xf32>) -> (tensor<?x?x?x?xf32>) attributes {entrance} {
    // CHECK: llh.reshape
    // CHECK-SAME: -> tensor<1x1x1x1xf32>
    %29 = "llh.add"(%arg0, %arg1) : (tensor<?x?x?x?xf32>, tensor<1xf32>) -> tensor<?x?x?x?xf32>
    return %29: tensor<?x?x?x?xf32>
  }
}

// -----
// CHECK-LABEL: where
func.func @where(%arg0: tensor<?x?x?x?xi1>,%arg1: tensor<1xf32>,%arg2: tensor<1xf32>) -> (tensor<?x?x?x?xf32>) attributes {entrance} {
    // CHECK: llh.reshape
    // CHECK-SAME: -> tensor<1x1x1x1xf32>
    // CHECK: llh.reshape
    // CHECK-SAME: -> tensor<1x1x1x1xf32>
    // CHECK: llh.broadcast_to
    // CHECK-SAME: -> tensor<?x?x?x?xf32>
    // CHECK: llh.broadcast_to
    // CHECK-SAME: -> tensor<?x?x?x?xf32>
    %0 = "llh.where"(%arg0, %arg1, %arg2) : (tensor<?x?x?x?xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?x?x?xf32>
    return %0: tensor<?x?x?x?xf32>
  }

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --canonicalize --cse /home/lfr/LLCompiler/test/Dialect/LLH/reshape_before_braodcast.mlir
