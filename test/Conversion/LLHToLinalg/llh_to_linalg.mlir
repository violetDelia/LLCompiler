// RUN: llc-opt --split-input-file --convert-llh-to-linalg %s| FileCheck %s

func.func @scalar_cast(%arg0: i64) ->(tensor<1xi64>) attributes {entrance}{
  // CHECK: tensor.empty
  // CHECK: linalg.fill
  %0 = "llh.scalar_cast"(%arg0) : (i64) -> tensor<1xi64>
  return %0: tensor<1xi64>
}

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-linalg /home/lfr/LLCompiler/test/Conversion/LLHToLinalg/llh_to_linalg.mlir
