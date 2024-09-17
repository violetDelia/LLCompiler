// -----// IR Dump After SimplifyAffineStructures (affine-simplify-structures) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_5 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %2 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %3 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %4 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %5 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %6 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %0, %alloc : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.for %arg1 = 0 to 1024 {
    %c0_17 = arith.constant 0 : index
    %8 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> (d0 floordiv 32)>(%arg1)
    %c0_18 = arith.constant 0 : index
    affine.store %cst, %alloc_6[%c0_18, %9, %8, %c0_17] : memref<1x32x32x1xf32>
  }
  %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_7 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc : memref<1x1x28x28xf32>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  affine.for %arg1 = 0 to 6272 {
    %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 28)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 28)>(%arg1)
    %c0_17 = arith.constant 0 : index
    affine.store %cst, %alloc_8[%c0_17, %10, %9, %8] : memref<1x28x28x8xf32>
  }
  affine.for %arg1 = 0 to 6272 {
    %c0_17 = arith.constant 0 : index
    %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 28)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 28)>(%arg1)
    %c0_18 = arith.constant 0 : index
    affine.store %cst, %alloc_8[%c0_3, %10, %9, %8] : memref<1x28x28x8xf32>
    affine.for %arg2 = 0 to 25 {
      %11 = affine.apply affine_map<(d0) -> (d0 mod 5)>(%arg2)
      %12 = affine.apply affine_map<(d0) -> (d0 floordiv 5)>(%arg2)
      %13 = affine.load %alloc_6[%c0_18, %10 + %12, %9 + %11, %c0_17] : memref<1x32x32x1xf32>
      %14 = affine.load %4[%8, %12, %11, %c0_17] : memref<8x5x5x1xf32>
      %15 = affine.load %alloc_8[%c0_18, %10, %9, %8] : memref<1x28x28x8xf32>
      %16 = arith.mulf %13, %14 : f32
      %17 = arith.addf %15, %16 : f32
      affine.store %17, %alloc_8[%c0_18, %10, %9, %8] : memref<1x28x28x8xf32>
    }
  }
  memref.dealloc %alloc_6 : memref<1x32x32x1xf32>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg1 = 0 to 1568 {
    %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 14)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 14)>(%arg1)
    %c0_17 = arith.constant 0 : index
    affine.store %cst_5, %alloc_9[%c0_4, %10, %9, %8] : memref<1x14x14x8xf32>
    affine.for %arg2 = 0 to 4 {
      %11 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg2)
      %12 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg2)
      %13 = affine.load %alloc_8[0, %10 * 2 + %12, %9 * 2 + %11, %8] : memref<1x28x28x8xf32>
      %14 = affine.load %3[0, %8, 0, 0] : memref<1x8x1x1xf32>
      %15 = affine.load %alloc_9[%c0_17, %10, %9, %8] : memref<1x14x14x8xf32>
      %16 = arith.addf %13, %14 : f32
      %17 = arith.maximumf %16, %cst : f32
      %18 = arith.maximumf %15, %17 : f32
      affine.store %18, %alloc_9[%c0_17, %10, %9, %8] : memref<1x14x14x8xf32>
    }
  }
  memref.dealloc %alloc_8 : memref<1x28x28x8xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg1 = 0 to 2592 {
    %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 18)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 18)>(%arg1)
    %c0_17 = arith.constant 0 : index
    affine.store %cst, %alloc_10[%c0_17, %10, %9, %8] : memref<1x18x18x8xf32>
  }
  %reinterpret_cast_11 = memref.reinterpret_cast %alloc_10 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_9, %reinterpret_cast_11 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_9 : memref<1x14x14x8xf32>
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  affine.for %arg1 = 0 to 3136 {
    %8 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 14)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) floordiv 14)>(%arg1)
    %c0_17 = arith.constant 0 : index
    affine.store %cst, %alloc_12[%c0, %10, %9, %8] : memref<1x14x14x16xf32>
    affine.for %arg2 = 0 to 200 {
      %11 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
      %12 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 5)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 5)>(%arg2)
      %14 = affine.load %alloc_10[%c0_17, %10 + %13, %9 + %12, %11] : memref<1x18x18x8xf32>
      %15 = affine.load %1[%8, %13, %12, %11] : memref<16x5x5x8xf32>
      %16 = affine.load %alloc_12[%c0_17, %10, %9, %8] : memref<1x14x14x16xf32>
      %17 = arith.mulf %14, %15 : f32
      %18 = arith.addf %16, %17 : f32
      affine.store %18, %alloc_12[%c0_17, %10, %9, %8] : memref<1x14x14x16xf32>
    }
  }
  memref.dealloc %alloc_10 : memref<1x18x18x8xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  affine.for %arg1 = 0 to 256 {
    %8 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 4)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) floordiv 4)>(%arg1)
    %c0_17 = arith.constant 0 : index
    affine.store %cst_5, %alloc_13[%c0_0, %10, %9, %8] : memref<1x4x4x16xf32>
    affine.for %arg2 = 0 to 9 {
      %11 = affine.apply affine_map<(d0) -> (d0 mod 3)>(%arg2)
      %12 = affine.apply affine_map<(d0) -> (d0 floordiv 3)>(%arg2)
      %13 = affine.load %alloc_12[0, %10 * 3 + %12, %9 * 3 + %11, %8] : memref<1x14x14x16xf32>
      %14 = affine.load %2[0, %8, 0, 0] : memref<1x16x1x1xf32>
      %15 = affine.load %alloc_13[%c0_17, %10, %9, %8] : memref<1x4x4x16xf32>
      %16 = arith.addf %13, %14 : f32
      %17 = arith.maximumf %16, %cst : f32
      %18 = arith.maximumf %15, %17 : f32
      affine.store %18, %alloc_13[%c0_17, %10, %9, %8] : memref<1x4x4x16xf32>
    }
  }
  memref.dealloc %alloc_12 : memref<1x14x14x16xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg1 = 0 to 256 {
    %8 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) mod 4)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) floordiv 4)>(%arg1)
    %c0_17 = arith.constant 0 : index
    %11 = affine.load %alloc_13[%c0_17, %9, %8, %10] : memref<1x4x4x16xf32>
    affine.store %11, %alloc_14[%c0_17, %10, %9, %8] : memref<1x16x4x4xf32>
  }
  memref.dealloc %alloc_13 : memref<1x4x4x16xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg1 = 0 to 10 {
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    affine.store %cst, %alloc_15[%c0_2, %c0_1, %arg1] : memref<1x1x10xf32>
    affine.for %arg2 = 0 to 256 {
      %8 = affine.load %alloc_14[symbol(%c0_18) + symbol(%c0_17), %arg2 floordiv 16, (%arg2 mod 16) floordiv 4, %arg2 mod 4] : memref<1x16x4x4xf32>
      %9 = affine.load %5[%c0_18, %arg2, %arg1] : memref<1x256x10xf32>
      %10 = affine.load %alloc_15[%c0_18, %c0_17, %arg1] : memref<1x1x10xf32>
      %11 = arith.mulf %8, %9 : f32
      %12 = arith.addf %10, %11 : f32
      affine.store %12, %alloc_15[%c0_18, %c0_17, %arg1] : memref<1x1x10xf32>
    }
  }
  memref.dealloc %alloc_14 : memref<1x16x4x4xf32>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg1 = 0 to 10 {
    %c0_17 = arith.constant 0 : index
    %8 = affine.load %alloc_15[0, 0, %arg1] : memref<1x1x10xf32>
    %9 = affine.load %6[0, %arg1] : memref<1x10xf32>
    %10 = arith.addf %8, %9 : f32
    affine.store %10, %alloc_16[%c0_17, %arg1] : memref<1x10xf32>
  }
  memref.dealloc %alloc_15 : memref<1x1x10xf32>
  %7 = bufferization.to_tensor %alloc_16 : memref<1x10xf32>
  memref.dealloc %alloc_16 : memref<1x10xf32>
  return %7 : tensor<1x10xf32>
}

