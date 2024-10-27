// RUN: llc-opt --split-input-file --fold-index-cast   %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --fold-index-cast  /home/lfr/LLCompiler/test/Dialect/IndexExtension/foldcast.mlir

func.func @cast() ->(index) attributes {entrance}{
    %c3 = arith.constant 3 : index
    %13 = index.castu %c3 : index to i64
    %17 = index.castu %13 : i64 to index
    // CHECK: arith.constant
    // CHECK-NOT : index.castu
    return %17 : index
}

// -----
func.func @index_const_to_arith() ->(index) attributes {entrance}{
    // CHECK: arith.constant
    // CHECK-NOT: index.constant
    %idx3 = index.constant 3
    return %idx3 : index
}

// -----
func.func @fold_tensor_dim(%arg0: tensor<?x?x?x?xf32>) ->(i64) attributes {entrance}{
    // CHECK: arith.constant
    // CHECK-NOT: index.constant
    %idx3 = index.constant 3
    %dim = tensor.dim {symbol = @s0} %arg0, %idx3 : tensor<?x?x?x?xf32>
    %2 = index.castu %dim : index to i64
    return %2 : i64
}

// -----

func.func @fold_from_elements(%arg0: tensor<?x?x?x?xf32>) ->(tensor<?x?x?x?xf32>) attributes {entrance}{
    %c3 = arith.constant 3 : i64
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim {symbol = @s0} %arg0, %c0 : tensor<?x?x?x?xf32>
    %0 = index.castu %dim : index to i64
    %dim_0 = tensor.dim {symbol = @s1} %arg0, %c1 : tensor<?x?x?x?xf32>
    %1 = index.castu %dim_0 : index to i64
    %dim_1 = tensor.dim {symbol = @s2} %arg0, %c2 : tensor<?x?x?x?xf32>
    %2 = index.castu %dim_1 : index to i64
    // CHECK: tensor.from_elements 
    // CHECK-SAME: tensor<4xindex>
    %from_elements = tensor.from_elements %0, %1, %2, %c3 : tensor<4xi64>
    %4 = "mhlo.dynamic_broadcast_in_dim"(%cst, %from_elements) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>, known_expanding_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>, known_nonexpanding_dimensions = dense<> : tensor<0xi64>}> : (tensor<1x1x1x1xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    return %4 : tensor<?x?x?x?xf32>
}