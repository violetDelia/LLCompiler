// RUN: llc-opt --split-input-file --preprocessing-llh-to-hlo %s| FileCheck %s

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --preprocessing-llh-to-hlo /home/lfr/LLCompiler/test/Conversion/LLHToHLO/preprocessing_llh_to_hlo.mlir

// CHECK-LABEL: relu
func.func @relu(%arg1: tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) ->() attributes {entrance}{
  // CHECK-NOT: relu
  // CHECK: llh.constant
  // CHECK: llh.max
  %102 = "llh.relu"(%arg1) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
  return 
}
