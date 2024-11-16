// RUN: llc-opt --split-input-file --infer-symbol-shape --canonicalize %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --canonicalize /home/lfr/LLCompiler/test/Dialect/LLH/canonicalize.mlir 


"llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: dim_to_const
func.func @dim_to_const(%101: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) ->(i64, i64, i64, i64) attributes {entrance} {
  // CHECK-COUNT-3: llh.constant
  %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
  // CHECK-COUNT-1: llh.dim
  %102 = "llh.dim"(%101, %2) <{symbol = @s0}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %103 = "llh.dim"(%101, %3) <{symbol = @c512}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %104 = "llh.dim"(%101, %1) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %105 = "llh.dim"(%101, %0) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  return %102,%103,%104,%105: i64, i64, i64, i64
}

// -----
"llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: fold_two_abs
func.func @fold_two_abs(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) ->  tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>> attributes {entrance} {
  // CHECK: llh.abs
  // CHECK: return
  %4 = "llh.abs"(%arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
  %5 = "llh.abs"(%4) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
  return %5 : tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
}

// -----
func.func @main(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s3"}) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = -1 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = -5 : i64}> : () -> i64
  // CHECK: llh.dim
  // CHECK: llh.extract
  %6 = "llh.extract"(%arg0, %0) : (tensor<?x?x?x?xf32>, i64) -> tensor<?x?x?xf32>
  // CHECK: llh.dim
  // CHECK: llh.sub
  // CHECK: llh.extract
  %7 = "llh.extract"(%arg0, %1) : (tensor<?x?x?x?xf32>, i64) -> tensor<?x?x?xf32>
  return 
}



