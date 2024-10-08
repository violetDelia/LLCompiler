// RUN: llc-opt --split-input-file --remove-redundant-ops %s| FileCheck %s

///home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --remove-redundant-ops /home/lfr/LLCompiler/test/Dialect/LLH/remove_redundant_ops.mlir

func.func @replaceFlattenOp(%arg2: tensor<?x512x1x1xf32>) ->() {
  // CHECK: %[[Reshape:.*]] = "llh.reshape"(%[[VAR0:.*]], %[[ND1:.*]], %[[ND2:.*]]) : (tensor<?x512x1x1xf32>, i64, i64) -> tensor<?x512xf32>
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %192 = "llh.flatten"(%arg2, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
  return 
}


// -----
#map1 = affine_map<()[s0, s1, s2] -> (s0, s1, s2, s1)>
module attributes {builtin.gloabal_layout = "NCHW"}{
  // CHECK-LABEL: replaceTorchSymbolicIntOp
  // CHECK-SAME: (%arg0: tensor<?x?x?x?xf32>)
  func.func @replaceTorchSymbolicIntOp(%arg0: tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>) ->() attributes {entrance}{
  // CHECK-NOT: llh.torch_symbolic_int
  // CHECK-NOT: llh.symbolic_bind
  %3 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
  %4 = "llh.torch_symbolic_int"() <{sym_name = "s1"}> : () -> i64
  %5 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
  %6 = "llh.reshape"(%arg0, %3, %5, %4, %5) : (tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>, i64, i64, i64, i64) -> tensor<?x?x?x?xf32>
  "llh.symbolic_bind"(%6, %3, %5, %4) <{expressions = #map1}> : (tensor<?x?x?x?xf32>, i64, i64, i64) -> ()
  return 
  }
}
