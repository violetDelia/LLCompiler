// RUN: llc-opt --split-input-file --convert-llh-to-shape %s| FileCheck %s
// 

// CHECK-LABEL: dim
func.func @dim(%arg0: tensor<?x3x?x?xf32>) ->() attributes {entrance}{
  %c3_i64 = arith.constant 3 : i64
  // CHECK-COUNT: index.casts
  // CHECK-COUNT: tensor.dim
  // CHECK: shape.dim
  %8 = "llh.dim"(%arg0, %c3_i64) : (tensor<?x3x?x?xf32>, i64) -> i64
  return 
}
