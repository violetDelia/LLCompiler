// RUN: llc-opt --split-input-file --convert-llh-to-math %s| FileCheck %s

// CHECK-LABEL: sqrt
func.func @sqrt(%arg0: i64) ->(i64) attributes {entrance}{
  // CHECK: arith.sitofp
  // CHECK: math.sqrt
  // CHECK: arith.fptosi
  %0 = "llh.sqrt"(%arg0) : (i64) -> i64
  return %0: i64
}

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-math /home/lfr/LLCompiler/test/Conversion/LLHToMath/llh_to_math.mlir
