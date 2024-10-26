func.func @conv2d_dyn_w_h(%input: tensor<?x100x100x3xf32>, %weights: tensor<1x7x7x3xf32>, %bias: tensor<1xf32>) -> (tensor<?x96x96x1xf32>) {
  %0 = tosa.conv2d %input, %weights, %bias {pad = array<i64: 1,1,1,1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<?x100x100x3xf32>, tensor<1x7x7x3xf32>, tensor<1xf32>) -> tensor<?x96x96x1xf32>
  return %0: tensor<?x96x96x1xf32>
}