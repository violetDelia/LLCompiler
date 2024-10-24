module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<1x3x7x7xf32> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "c7", func.input_symbol_3 = "c7"}, %arg1: tensor<?x3x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}) -> (tensor<?x1x?x?xf32>, tensor<1x3x7x7xf32>, tensor<?x3x?x?xf32>, i64, i64) attributes {entrance} {
    %c0_i64 = arith.constant {symbol = @c0} 0 : i64
    %c2_i64 = arith.constant {symbol = @c2} 2 : i64
    %0 = index.castu %c0_i64 : i64 to index
    %dim = tensor.dim {symbol = @s0} %arg1, %0 : tensor<?x3x?x?xf32>
    %1 = index.castu %dim : index to i64
    %2 = index.castu %c2_i64 : i64 to index
    %dim_0 = tensor.dim {symbol = @s2} %arg1, %2 : tensor<?x3x?x?xf32>
    %3 = index.castu %dim_0 : index to i64
    %4 = stablehlo.transpose %arg1, dims = [0, 2, 3, 1] : (tensor<?x3x?x?xf32>) -> tensor<?x?x?x3xf32>
    %5 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<1x3x7x7xf32>) -> tensor<1x7x7x3xf32>
    %6 = stablehlo.convolution(%4, %5) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x?x?x3xf32>, tensor<1x7x7x3xf32>) -> tensor<?x?x?x1xf32>
    %7 = stablehlo.transpose %6, dims = [0, 3, 1, 2] : (tensor<?x?x?x1xf32>) -> tensor<?x1x?x?xf32>
    return %7, %arg0, %arg1, %1, %3 : tensor<?x1x?x?xf32>, tensor<1x3x7x7xf32>, tensor<?x3x?x?xf32>, i64, i64
  }
  module @__symbol__ {
  }
}