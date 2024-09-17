// -----// IR Dump After SimplifyAffineStructures (affine-simplify-structures) //----- //
func.func @CNTKGraph(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x10xf32>) {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_5 = arith.constant -3.40282347E+38 : f32
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %c1024 = arith.constant 1024 : index
  affine.parallel (%arg2) = (0) to (symbol(%c1024)) {
    %c0_20 = arith.constant 0 : index
    %6 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 32)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) floordiv 32)>(%arg2)
    affine.store %cst, %alloc_6[%8, %7, %6, %c0_20] : memref<1x32x32x1xf32>
  }
  %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_7 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc : memref<1x1x28x28xf32>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  %c6272 = arith.constant 6272 : index
  affine.parallel (%arg2) = (0) to (symbol(%c6272)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 28)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 28) mod 28)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 28) floordiv 28)>(%arg2)
    affine.store %cst, %alloc_8[%9, %8, %7, %6] : memref<1x28x28x8xf32>
  }
  %c6272_9 = arith.constant 6272 : index
  affine.parallel (%arg2) = (0) to (symbol(%c6272_9)) {
    %c0_20 = arith.constant 0 : index
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 28)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 28) mod 28)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 28) floordiv 28)>(%arg2)
    affine.store %cst, %alloc_8[%c0_2, %8, %7, %6] : memref<1x28x28x8xf32>
    affine.for %arg3 = 0 to 25 {
      %10 = affine.apply affine_map<(d0) -> (d0 mod 5)>(%arg3)
      %11 = affine.apply affine_map<(d0) -> (d0 floordiv 5)>(%arg3)
      %12 = affine.load %alloc_6[%9, %8 + %11, %7 + %10, %c0_20] : memref<1x32x32x1xf32>
      %13 = affine.load %3[%6, %11, %10, %c0_20] : memref<8x5x5x1xf32>
      %14 = affine.load %alloc_8[%9, %8, %7, %6] : memref<1x28x28x8xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      affine.store %16, %alloc_8[%9, %8, %7, %6] : memref<1x28x28x8xf32>
    }
  }
  memref.dealloc %alloc_6 : memref<1x32x32x1xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %c1568 = arith.constant 1568 : index
  affine.parallel (%arg2) = (0) to (symbol(%c1568)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 14)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 14) mod 14)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 14) floordiv 14)>(%arg2)
    affine.store %cst_5, %alloc_10[%c0_4, %8, %7, %6] : memref<1x14x14x8xf32>
    affine.for %arg3 = 0 to 4 {
      %10 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg3)
      %11 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg3)
      %12 = affine.load %alloc_8[0, %8 * 2 + %11, %7 * 2 + %10, %6] : memref<1x28x28x8xf32>
      %13 = affine.load %2[0, %6, 0, 0] : memref<1x8x1x1xf32>
      %14 = affine.load %alloc_10[%9, %8, %7, %6] : memref<1x14x14x8xf32>
      %15 = arith.addf %12, %13 : f32
      %16 = arith.maximumf %15, %cst : f32
      %17 = arith.maximumf %14, %16 : f32
      affine.store %17, %alloc_10[%9, %8, %7, %6] : memref<1x14x14x8xf32>
    }
  }
  memref.dealloc %alloc_8 : memref<1x28x28x8xf32>
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %c2592 = arith.constant 2592 : index
  affine.parallel (%arg2) = (0) to (symbol(%c2592)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 18)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 18) mod 18)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 18) floordiv 18)>(%arg2)
    affine.store %cst, %alloc_11[%9, %8, %7, %6] : memref<1x18x18x8xf32>
  }
  %reinterpret_cast_12 = memref.reinterpret_cast %alloc_11 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_10, %reinterpret_cast_12 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_10 : memref<1x14x14x8xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  %c3136 = arith.constant 3136 : index
  affine.parallel (%arg2) = (0) to (symbol(%c3136)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 14)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 16) floordiv 14) mod 14)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 16) floordiv 14) floordiv 14)>(%arg2)
    affine.store %cst, %alloc_13[%c0_3, %8, %7, %6] : memref<1x14x14x16xf32>
    affine.for %arg3 = 0 to 200 {
      %10 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg3)
      %11 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 5)>(%arg3)
      %12 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 5)>(%arg3)
      %13 = affine.load %alloc_11[%9, %8 + %12, %7 + %11, %10] : memref<1x18x18x8xf32>
      %14 = affine.load %0[%6, %12, %11, %10] : memref<16x5x5x8xf32>
      %15 = affine.load %alloc_13[%9, %8, %7, %6] : memref<1x14x14x16xf32>
      %16 = arith.mulf %13, %14 : f32
      %17 = arith.addf %15, %16 : f32
      affine.store %17, %alloc_13[%9, %8, %7, %6] : memref<1x14x14x16xf32>
    }
  }
  memref.dealloc %alloc_11 : memref<1x18x18x8xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  %c256 = arith.constant 256 : index
  affine.parallel (%arg2) = (0) to (symbol(%c256)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 4)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 16) floordiv 4) mod 4)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 16) floordiv 4) floordiv 4)>(%arg2)
    affine.store %cst_5, %alloc_14[%c0, %8, %7, %6] : memref<1x4x4x16xf32>
    affine.for %arg3 = 0 to 9 {
      %10 = affine.apply affine_map<(d0) -> (d0 mod 3)>(%arg3)
      %11 = affine.apply affine_map<(d0) -> (d0 floordiv 3)>(%arg3)
      %12 = affine.load %alloc_13[0, %8 * 3 + %11, %7 * 3 + %10, %6] : memref<1x14x14x16xf32>
      %13 = affine.load %1[0, %6, 0, 0] : memref<1x16x1x1xf32>
      %14 = affine.load %alloc_14[%9, %8, %7, %6] : memref<1x4x4x16xf32>
      %15 = arith.addf %12, %13 : f32
      %16 = arith.maximumf %15, %cst : f32
      %17 = arith.maximumf %14, %16 : f32
      affine.store %17, %alloc_14[%9, %8, %7, %6] : memref<1x4x4x16xf32>
    }
  }
  memref.dealloc %alloc_13 : memref<1x14x14x16xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %c256_16 = arith.constant 256 : index
  affine.parallel (%arg2) = (0) to (symbol(%c256_16)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) mod 4)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 4) floordiv 4) mod 16)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 4) floordiv 4) floordiv 16)>(%arg2)
    %10 = affine.load %alloc_14[%9, %7, %6, %8] : memref<1x4x4x16xf32>
    affine.store %10, %alloc_15[%9, %8, %7, %6] : memref<1x16x4x4xf32>
  }
  memref.dealloc %alloc_14 : memref<1x4x4x16xf32>
  %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %c10 = arith.constant 10 : index
  affine.parallel (%arg2) = (0) to (symbol(%c10)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%arg2)
    %c0_20 = arith.constant 0 : index
    %7 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg2)
    affine.store %cst, %alloc_17[%c0_1, %c0_0, %6] : memref<1x1x10xf32>
    affine.for %arg3 = 0 to 256 {
      %8 = affine.load %alloc_15[symbol(%7) + symbol(%c0_20), %arg3 floordiv 16, (%arg3 mod 16) floordiv 4, %arg3 mod 4] : memref<1x16x4x4xf32>
      %9 = affine.load %4[%7, %arg3, %6] : memref<1x256x10xf32>
      %10 = affine.load %alloc_17[%7, %c0_20, %6] : memref<1x1x10xf32>
      %11 = arith.mulf %8, %9 : f32
      %12 = arith.addf %10, %11 : f32
      affine.store %12, %alloc_17[%7, %c0_20, %6] : memref<1x1x10xf32>
    }
  }
  memref.dealloc %alloc_15 : memref<1x16x4x4xf32>
  %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %c10_19 = arith.constant 10 : index
  affine.parallel (%arg2) = (0) to (symbol(%c10_19)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg2)
    %8 = affine.load %alloc_17[0, 0, %6] : memref<1x1x10xf32>
    %9 = affine.load %5[0, %6] : memref<1x10xf32>
    %10 = arith.addf %8, %9 : f32
    affine.store %10, %alloc_18[%7, %6] : memref<1x10xf32>
  }
  memref.dealloc %alloc_17 : memref<1x1x10xf32>
  memref.copy %alloc_18, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  memref.dealloc %alloc_18 : memref<1x10xf32>
  return
}

