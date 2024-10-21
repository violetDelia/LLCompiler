#map = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 + 10)>
module {
  func.func @dim(%arg0: tensor<*xf32>, %arg1: index) -> index {
    %0 = bufferization.to_memref %arg0 : memref<*xf32>
    %dim = memref.dim %0, %arg1 : memref<*xf32>
    return %dim : index
  }
  func.func @rank(%arg0: tensor<*xf32>) -> index {
    %0 = bufferization.to_memref %arg0 : memref<*xf32>
    %1 = memref.rank %0 : memref<*xf32>
    return %1 : index
  }
  func.func @tensor.cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
    %0 = bufferization.to_memref %arg0 : memref<?xindex>
    %cast = memref.cast %0 : memref<?xindex> to memref<2xindex>
    %1 = bufferization.to_tensor %cast : memref<2xindex>
    return %1 : tensor<2xindex>
  }
  func.func @tensor.cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
    %0 = bufferization.to_memref %arg0 : memref<*xf32>
    %cast = memref.cast %0 : memref<*xf32> to memref<2xf32, strided<[?], offset: ?>>
    %1 = bufferization.to_tensor %cast : memref<2xf32, strided<[?], offset: ?>>
    return %1 : tensor<2xf32>
  }
  func.func @tensor.cast_to_unranked(%arg0: tensor<2xf32>) -> tensor<*xf32> {
    %0 = bufferization.to_memref %arg0 : memref<2xf32>
    %cast = memref.cast %0 : memref<2xf32> to memref<*xf32>
    %1 = bufferization.to_tensor %cast : memref<*xf32>
    return %1 : tensor<*xf32>
  }
  func.func @tensor.empty() -> tensor<5xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5xf32>
    %0 = bufferization.to_tensor %alloc : memref<5xf32>
    return %0 : tensor<5xf32>
  }
  func.func @tensor.extract(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
    %0 = bufferization.to_memref %arg0 : memref<?xf32>
    %1 = memref.load %0[%arg1] : memref<?xf32>
    return %1 : f32
  }
  func.func @tensor.from_elements_0d(%arg0: index) -> tensor<index> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<index>
    memref.store %arg0, %alloc[] : memref<index>
    %0 = bufferization.to_tensor %alloc : memref<index>
    return %0 : tensor<index>
  }
  func.func @tensor.from_elements_1d(%arg0: index, %arg1: index) -> tensor<2xindex> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %arg0, %alloc[%c0] : memref<2xindex>
    memref.store %arg1, %alloc[%c1] : memref<2xindex>
    %0 = bufferization.to_tensor %alloc : memref<2xindex>
    return %0 : tensor<2xindex>
  }
  func.func @tensor.from_elements_2d(%arg0: index, %arg1: index) -> tensor<3x2xindex> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x2xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %arg0, %alloc[%c0, %c0] : memref<3x2xindex>
    memref.store %arg1, %alloc[%c0, %c1] : memref<3x2xindex>
    memref.store %arg0, %alloc[%c1, %c0] : memref<3x2xindex>
    memref.store %arg1, %alloc[%c1, %c1] : memref<3x2xindex>
    memref.store %arg0, %alloc[%c2, %c0] : memref<3x2xindex>
    memref.store %arg1, %alloc[%c2, %c1] : memref<3x2xindex>
    %0 = bufferization.to_tensor %alloc : memref<3x2xindex>
    return %0 : tensor<3x2xindex>
  }
  func.func @tensor.from_elements_3d(%arg0: f32) -> tensor<3x2x2xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 4.000000e+00 : f32
    %cst_3 = arith.constant 5.000000e+00 : f32
    %cst_4 = arith.constant 6.000000e+00 : f32
    %cst_5 = arith.constant 7.000000e+00 : f32
    %cst_6 = arith.constant 8.000000e+00 : f32
    %cst_7 = arith.constant 9.000000e+00 : f32
    %cst_8 = arith.constant 1.000000e+01 : f32
    %cst_9 = arith.constant 1.100000e+01 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x2x2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %arg0, %alloc[%c0, %c0, %c0] : memref<3x2x2xf32>
    memref.store %cst, %alloc[%c0, %c0, %c1] : memref<3x2x2xf32>
    memref.store %cst_0, %alloc[%c0, %c1, %c0] : memref<3x2x2xf32>
    memref.store %cst_1, %alloc[%c0, %c1, %c1] : memref<3x2x2xf32>
    memref.store %cst_2, %alloc[%c1, %c0, %c0] : memref<3x2x2xf32>
    memref.store %cst_3, %alloc[%c1, %c0, %c1] : memref<3x2x2xf32>
    memref.store %cst_4, %alloc[%c1, %c1, %c0] : memref<3x2x2xf32>
    memref.store %cst_5, %alloc[%c1, %c1, %c1] : memref<3x2x2xf32>
    memref.store %cst_6, %alloc[%c2, %c0, %c0] : memref<3x2x2xf32>
    memref.store %cst_7, %alloc[%c2, %c0, %c1] : memref<3x2x2xf32>
    memref.store %cst_8, %alloc[%c2, %c1, %c0] : memref<3x2x2xf32>
    memref.store %cst_9, %alloc[%c2, %c1, %c1] : memref<3x2x2xf32>
    %0 = bufferization.to_tensor %alloc : memref<3x2x2xf32>
    return %0 : tensor<3x2x2xf32>
  }
  func.func @tensor.generate(%arg0: tensor<*xf32>, %arg1: index) -> tensor<?xindex> {
    %0 = bufferization.to_memref %arg0 : memref<*xf32>
    %alloc = memref.alloc(%arg1) {alignment = 64 : i64} : memref<?xindex>
    %1 = bufferization.to_tensor %alloc : memref<?xindex>
    %mapped = linalg.map outs(%1 : tensor<?xindex>)
      () {
        %2 = linalg.index 0 : index
        %dim = memref.dim %0, %2 : memref<*xf32>
        linalg.yield %dim : index
      }
    return %mapped : tensor<?xindex>
  }
  func.func @tensor.generate_static_and_dynamic(%arg0: index) -> tensor<16x?xindex> {
    %alloc = memref.alloc(%arg0) {alignment = 64 : i64} : memref<16x?xindex>
    %0 = bufferization.to_tensor %alloc : memref<16x?xindex>
    %mapped = linalg.map outs(%0 : tensor<16x?xindex>)
      () {
        %1 = linalg.index 0 : index
        %2 = linalg.index 1 : index
        %3 = arith.addi %1, %2 : index
        linalg.yield %3 : index
      }
    return %mapped : tensor<16x?xindex>
  }
  func.func @tensor.generate_unknown_ops_in_body(%arg0: index) -> tensor<?xindex> {
    %alloc = memref.alloc(%arg0) {alignment = 64 : i64} : memref<?xindex>
    %0 = bufferization.to_tensor %alloc : memref<?xindex>
    %mapped = linalg.map outs(%0 : tensor<?xindex>)
      () {
        %1 = linalg.index 0 : index
        %2 = "test.source"() : () -> index
        linalg.yield %2 : index
      }
    return %mapped : tensor<?xindex>
  }
  func.func @tensor.extract_slice(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x10xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %subview = memref.subview %0[5, %arg2] [%arg1, 10] [1, 1] : memref<?x?xf32> to memref<?x10xf32, strided<[?, 1], offset: ?>>
    %1 = bufferization.to_tensor %subview : memref<?x10xf32, strided<[?, 1], offset: ?>>
    return %1 : tensor<?x10xf32>
  }
  func.func @tensor.extract_slice_rank_reducing(%arg0: tensor<?x10x?xf32>, %arg1: index, %arg2: index) -> tensor<?x15xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?x10x?xf32>
    %subview = memref.subview %0[5, %arg1, 10] [%arg2, 1, 15] [1, 1, 1] : memref<?x10x?xf32> to memref<?x15xf32, strided<[?, 1], offset: ?>>
    %1 = bufferization.to_tensor %subview : memref<?x15xf32, strided<[?, 1], offset: ?>>
    return %1 : tensor<?x15xf32>
  }
  func.func @tensor.insert_slice(%arg0: tensor<?x?xf32>, %arg1: tensor<?x10xf32>, %arg2: index, %arg3: index) -> tensor<?x?xf32> {
    %0 = bufferization.to_memref %arg1 : memref<?x10xf32>
    %1 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %1, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %1, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    memref.copy %1, %alloc : memref<?x?xf32> to memref<?x?xf32>
    %subview = memref.subview %alloc[%arg2, 5] [%arg3, 10] [1, 1] : memref<?x?xf32> to memref<?x10xf32, strided<[?, 1], offset: ?>>
    memref.copy %0, %subview : memref<?x10xf32> to memref<?x10xf32, strided<[?, 1], offset: ?>>
    %2 = bufferization.to_tensor %alloc : memref<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
  func.func @tensor.insert_slice_rank_reducing_1(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: index, %arg3: index) -> tensor<?x?xf32> {
    %0 = bufferization.to_memref %arg1 : memref<f32>
    %1 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %1, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %1, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    memref.copy %1, %alloc : memref<?x?xf32> to memref<?x?xf32>
    %subview = memref.subview %alloc[%arg2, %arg3] [1, 1] [1, 1] : memref<?x?xf32> to memref<f32, strided<[], offset: ?>>
    memref.copy %0, %subview : memref<f32> to memref<f32, strided<[], offset: ?>>
    %2 = bufferization.to_tensor %alloc : memref<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
  func.func @tensor.insert_slice_rank_reducing_2(%arg0: tensor<?x?x?x?x?x?x?xf32>, %arg1: tensor<2x1x4x1x1xf32>, %arg2: index) -> tensor<?x?x?x?x?x?x?xf32> {
    %0 = bufferization.to_memref %arg1 : memref<2x1x4x1x1xf32>
    %1 = bufferization.to_memref %arg0 : memref<?x?x?x?x?x?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %1, %c0 : memref<?x?x?x?x?x?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %1, %c1 : memref<?x?x?x?x?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_1 = memref.dim %1, %c2 : memref<?x?x?x?x?x?x?xf32>
    %c3 = arith.constant 3 : index
    %dim_2 = memref.dim %1, %c3 : memref<?x?x?x?x?x?x?xf32>
    %c4 = arith.constant 4 : index
    %dim_3 = memref.dim %1, %c4 : memref<?x?x?x?x?x?x?xf32>
    %c5 = arith.constant 5 : index
    %dim_4 = memref.dim %1, %c5 : memref<?x?x?x?x?x?x?xf32>
    %c6 = arith.constant 6 : index
    %dim_5 = memref.dim %1, %c6 : memref<?x?x?x?x?x?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0, %dim_1, %dim_2, %dim_3, %dim_4, %dim_5) {alignment = 64 : i64} : memref<?x?x?x?x?x?x?xf32>
    memref.copy %1, %alloc : memref<?x?x?x?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>
    %subview = memref.subview %alloc[%arg2, %arg2, %arg2, %arg2, %arg2, %arg2, %arg2] [1, 2, 1, 4, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1] : memref<?x?x?x?x?x?x?xf32> to memref<2x1x4x1x1xf32, strided<[?, ?, ?, ?, ?], offset: ?>>
    memref.copy %0, %subview : memref<2x1x4x1x1xf32> to memref<2x1x4x1x1xf32, strided<[?, ?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_tensor %alloc : memref<?x?x?x?x?x?x?xf32>
    return %2 : tensor<?x?x?x?x?x?x?xf32>
  }
  func.func @tensor.insert(%arg0: tensor<5xf32>, %arg1: index, %arg2: f32) -> tensor<5xf32> {
    %0 = bufferization.to_memref %arg0 : memref<5xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5xf32>
    memref.copy %0, %alloc : memref<5xf32> to memref<5xf32>
    memref.store %arg2, %alloc[%arg1] : memref<5xf32>
    %1 = bufferization.to_tensor %alloc : memref<5xf32>
    return %1 : tensor<5xf32>
  }
  func.func @tensor.expand_shape(%arg0: tensor<?x10xf32>, %arg1: index) -> tensor<2x?x10xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?x10xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %0, %c0 : memref<?x10xf32>
    %c2 = arith.constant 2 : index
    %1 = arith.divui %dim, %c2 : index
    %expand_shape = memref.expand_shape %0 [[0, 1], [2]] output_shape [2, %1, 10] : memref<?x10xf32> into memref<2x?x10xf32>
    %2 = bufferization.to_tensor %expand_shape : memref<2x?x10xf32>
    return %2 : tensor<2x?x10xf32>
  }
  func.func @tensor.expand_shape_of_slice(%arg0: tensor<?x20xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<?x7x2x5xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?x20xf32>
    %subview = memref.subview %0[%arg1, 5] [%arg2, 10] [1, 1] : memref<?x20xf32> to memref<?x10xf32, strided<[20, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %1 = arith.divui %arg2, %c7 : index
    %expand_shape = memref.expand_shape %subview [[0, 1], [2, 3]] output_shape [%1, 7, 2, 5] : memref<?x10xf32, strided<[20, 1], offset: ?>> into memref<?x7x2x5xf32, strided<[140, 20, 5, 1], offset: ?>>
    %2 = bufferization.to_tensor %expand_shape : memref<?x7x2x5xf32, strided<[140, 20, 5, 1], offset: ?>>
    return %2 : tensor<?x7x2x5xf32>
  }
  func.func @tensor.expand_shape_of_scalar_slice(%arg0: tensor<?xf32>, %arg1: index, %arg2: index) -> tensor<1xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?xf32>
    %subview = memref.subview %0[%arg1] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
    %expand_shape = memref.expand_shape %subview [] output_shape [1] : memref<f32, strided<[], offset: ?>> into memref<1xf32, strided<[1], offset: ?>>
    %1 = bufferization.to_tensor %expand_shape : memref<1xf32, strided<[1], offset: ?>>
    return %1 : tensor<1xf32>
  }
  func.func @tensor.collapse_shape(%arg0: tensor<2x?x?xf32>) -> tensor<?x?xf32> {
    %0 = bufferization.to_memref %arg0 : memref<2x?x?xf32>
    %collapse_shape = memref.collapse_shape %0 [[0, 1], [2]] : memref<2x?x?xf32> into memref<?x?xf32>
    %1 = bufferization.to_tensor %collapse_shape : memref<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
  func.func @tensor.collapse_shape_to_scalar(%arg0: tensor<1x1x1xf32>) -> tensor<f32> {
    %0 = bufferization.to_memref %arg0 : memref<1x1x1xf32>
    %collapse_shape = memref.collapse_shape %0 [] : memref<1x1x1xf32> into memref<f32>
    %1 = bufferization.to_tensor %collapse_shape : memref<f32>
    return %1 : tensor<f32>
  }
  func.func @tensor.collapse_shape_of_slice(%arg0: tensor<2xi32>) -> tensor<i32> {
    %0 = bufferization.to_memref %arg0 : memref<2xi32>
    %subview = memref.subview %0[1] [1] [1] : memref<2xi32> to memref<1xi32, strided<[1], offset: 1>>
    %collapse_shape = memref.collapse_shape %subview [] : memref<1xi32, strided<[1], offset: 1>> into memref<i32, strided<[], offset: 1>>
    %1 = bufferization.to_tensor %collapse_shape : memref<i32, strided<[], offset: 1>>
    return %1 : tensor<i32>
  }
  func.func @tensor.collapse_shape_of_slice2(%arg0: tensor<?x?x?x?xi64>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<87x63648xi64> {
    %0 = bufferization.to_memref %arg0 : memref<?x?x?x?xi64>
    %subview = memref.subview %0[%arg1, %arg2, %arg3, %arg4] [87, 78, 68, 12] [1, 1, 1, 1] : memref<?x?x?x?xi64> to memref<87x78x68x12xi64, strided<[?, ?, ?, 1], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<87x78x68x12xi64>
    memref.copy %subview, %alloc : memref<87x78x68x12xi64, strided<[?, ?, ?, 1], offset: ?>> to memref<87x78x68x12xi64>
    %collapse_shape = memref.collapse_shape %alloc [[0], [1, 2, 3]] : memref<87x78x68x12xi64> into memref<87x63648xi64>
    %1 = bufferization.to_tensor %collapse_shape : memref<87x63648xi64>
    return %1 : tensor<87x63648xi64>
  }
  func.func @tensor.collapse_shape_of_slice3(%arg0: tensor<1x2xf32>) -> tensor<1xf32> {
    %0 = bufferization.to_memref %arg0 : memref<1x2xf32>
    %subview = memref.subview %0[0, 0] [1, 1] [1, 1] : memref<1x2xf32> to memref<1x1xf32, strided<[2, 1]>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<1x1xf32, strided<[2, 1]>> into memref<1xf32, strided<[2]>>
    %1 = bufferization.to_tensor %collapse_shape : memref<1xf32, strided<[2]>>
    return %1 : tensor<1xf32>
  }
  func.func @tensor.collapse_shape_of_slice4(%arg0: tensor<?x2x4xf32>, %arg1: index, %arg2: index) -> tensor<8xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?x2x4xf32>
    %subview = memref.subview %0[0, 0, %arg1] [4, 2, 1] [1, 1, 1] : memref<?x2x4xf32> to memref<4x2x1xf32, strided<[8, 4, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1, 2]] : memref<4x2x1xf32, strided<[8, 4, 1], offset: ?>> into memref<8xf32, strided<[4], offset: ?>>
    %1 = bufferization.to_tensor %collapse_shape : memref<8xf32, strided<[4], offset: ?>>
    return %1 : tensor<8xf32>
  }
  func.func @tensor.collapse_shape_of_slice5(%arg0: tensor<2x2x2xi64>) -> tensor<4xi64> {
    %0 = bufferization.to_memref %arg0 : memref<2x2x2xi64>
    %subview = memref.subview %0[0, 0, 0] [2, 1, 2] [1, 1, 1] : memref<2x2x2xi64> to memref<2x1x2xi64, strided<[4, 2, 1]>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x1x2xi64>
    memref.copy %subview, %alloc : memref<2x1x2xi64, strided<[4, 2, 1]>> to memref<2x1x2xi64>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1, 2]] : memref<2x1x2xi64> into memref<4xi64>
    %1 = bufferization.to_tensor %collapse_shape : memref<4xi64>
    return %1 : tensor<4xi64>
  }
  func.func @tensor.reshape(%arg0: tensor<?x10xf32>) -> tensor<2x2x5xf32> {
    %0 = bufferization.to_memref %arg0 : memref<?x10xf32>
    %c2_i64 = arith.constant 2 : i64
    %c5_i64 = arith.constant 5 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %c2_i64, %alloc[%c0] : memref<3xi64>
    memref.store %c2_i64, %alloc[%c1] : memref<3xi64>
    memref.store %c5_i64, %alloc[%c2] : memref<3xi64>
    %reshape = memref.reshape %0(%alloc) : (memref<?x10xf32>, memref<3xi64>) -> memref<2x2x5xf32>
    %1 = bufferization.to_tensor %reshape : memref<2x2x5xf32>
    return %1 : tensor<2x2x5xf32>
  }
  func.func @tensor.pad(%arg0: tensor<?x10xindex>, %arg1: index, %arg2: index, %arg3: index) -> tensor<?x?xindex> {
    %0 = bufferization.to_memref %arg0 : memref<?x10xindex>
    %1 = bufferization.to_memref %arg0 : memref<?x10xindex>
    %2 = bufferization.to_memref %arg0 : memref<?x10xindex>
    %3 = bufferization.to_memref %arg0 : memref<?x10xindex>
    %4 = bufferization.to_memref %arg0 : memref<?x10xindex>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %4, %c0 : memref<?x10xindex>
    %c5 = arith.constant 5 : index
    %5 = affine.apply #map()[%dim, %c5, %arg2]
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %3, %c1 : memref<?x10xindex>
    %6 = affine.apply #map()[%dim_0, %arg1, %arg3]
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %2, %c0_1 : memref<?x10xindex>
    %7 = affine.apply #map1()[%arg2, %dim_2]
    %c1_3 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %8 = affine.apply #map2()[%arg1, %arg3]
    %alloc = memref.alloc(%7, %8) {alignment = 64 : i64} : memref<?x?xindex>
    %9 = bufferization.to_tensor %alloc : memref<?x?xindex>
    %mapped = linalg.map outs(%9 : tensor<?x?xindex>)
      () {
        %12 = linalg.index 0 : index
        %13 = linalg.index 1 : index
        %14 = arith.muli %12, %13 : index
        linalg.yield %14 : index
      }
    %10 = bufferization.to_memref %mapped : memref<?x?xindex>
    %c0_4 = arith.constant 0 : index
    %dim_5 = memref.dim %1, %c0_4 : memref<?x10xindex>
    %subview = memref.subview %10[5, %arg1] [%dim_5, 10] [1, 1] : memref<?x?xindex> to memref<?x10xindex, strided<[?, 1], offset: ?>>
    memref.copy %0, %subview : memref<?x10xindex> to memref<?x10xindex, strided<[?, 1], offset: ?>>
    %11 = bufferization.to_tensor %10 : memref<?x?xindex>
    return %11 : tensor<?x?xindex>
  }
  func.func @tensor.splat(%arg0: f32) -> tensor<10x2x4xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x2x4xf32>
    %0 = bufferization.to_tensor %alloc : memref<10x2x4xf32>
    %mapped = linalg.map outs(%0 : tensor<10x2x4xf32>)
      () {
        linalg.yield %arg0 : f32
      }
    return %mapped : tensor<10x2x4xf32>
  }
  func.func @tensor.splat_dynamic(%arg0: f32, %arg1: index, %arg2: index) -> tensor<?x3x?xf32> {
    %alloc = memref.alloc(%arg1, %arg2) {alignment = 64 : i64} : memref<?x3x?xf32>
    %0 = bufferization.to_tensor %alloc : memref<?x3x?xf32>
    %mapped = linalg.map outs(%0 : tensor<?x3x?xf32>)
      () {
        linalg.yield %arg0 : f32
      }
    return %mapped : tensor<?x3x?xf32>
  }
  func.func @parallel_insert_slice_copy_before_write(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
    %0 = bufferization.to_memref %arg0 : memref<4xf32>
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %1 = scf.forall (%arg2) in (%c4) shared_outs(%arg3 = %arg1) -> (tensor<4xf32>) {
      %2 = bufferization.to_memref %arg3 : memref<4xf32>
      %subview = memref.subview %0[%arg2] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %2[%arg2] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_0 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
      scf.forall.in_parallel {
      }
    }
    return
  }
}

