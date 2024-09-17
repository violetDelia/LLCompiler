// -----// IR Dump After AffineScalarReplacement (affine-scalrep) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %alloc_2 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %c0_5 = arith.constant 0 : index
  %c0_6 = arith.constant 0 : index
  %c0_7 = arith.constant 0 : index
  %c0_8 = arith.constant 0 : index
  %alloc_9 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_10 = arith.constant 0 : index
  %c0_11 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_12 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %2 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %3 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %4 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %5 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %6 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %0, %alloc_13 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc_13 to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst, %alloc_14[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %reinterpret_cast_15 = memref.reinterpret_cast %alloc_14 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_15 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 14 {
      affine.for %arg3 = 0 to 14 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_12, %alloc_16[%c0_6, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 2 {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg2, %arg5)
              %9 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              affine.store %cst, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %10 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg2, %arg5)
              %11 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              affine.for %arg7 = 0 to 5 {
                %22 = affine.load %alloc_14[%c0_5, %10 + %arg7, %11 + %c0, %c0_4] : memref<1x32x32x1xf32>
                %23 = affine.load %4[%arg4, %arg7, %c0, %c0_4] : memref<8x5x5x1xf32>
                %24 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                %25 = arith.mulf %22, %23 : f32
                %26 = arith.addf %24, %25 : f32
                affine.store %26, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                %27 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c0)
                %28 = affine.load %alloc_14[%c0_5, %10 + %arg7, %11 + %27, %c0_4] : memref<1x32x32x1xf32>
                %29 = affine.load %4[%arg4, %arg7, %27, %c0_4] : memref<8x5x5x1xf32>
                %30 = arith.mulf %28, %29 : f32
                %31 = arith.addf %26, %30 : f32
                affine.store %31, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                %32 = affine.apply affine_map<(d0) -> (d0 + 2)>(%c0)
                %33 = affine.load %alloc_14[%c0_5, %10 + %arg7, %11 + %32, %c0_4] : memref<1x32x32x1xf32>
                %34 = affine.load %4[%arg4, %arg7, %32, %c0_4] : memref<8x5x5x1xf32>
                %35 = arith.mulf %33, %34 : f32
                %36 = arith.addf %31, %35 : f32
                affine.store %36, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                %37 = affine.apply affine_map<(d0) -> (d0 + 3)>(%c0)
                %38 = affine.load %alloc_14[%c0_5, %10 + %arg7, %11 + %37, %c0_4] : memref<1x32x32x1xf32>
                %39 = affine.load %4[%arg4, %arg7, %37, %c0_4] : memref<8x5x5x1xf32>
                %40 = arith.mulf %38, %39 : f32
                %41 = arith.addf %36, %40 : f32
                affine.store %41, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                %42 = affine.load %alloc_14[%c0_5, %10 + %arg7, %11 + %c4, %c0_4] : memref<1x32x32x1xf32>
                %43 = affine.load %4[%arg4, %arg7, %c4, %c0_4] : memref<8x5x5x1xf32>
                %44 = arith.mulf %42, %43 : f32
                %45 = arith.addf %41, %44 : f32
                affine.store %45, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
              }
              %12 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (0)>(%arg4, %arg2, %arg5, %arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2)>(%arg4, %arg2, %arg5, %arg3, %arg6)
              %14 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d3 * 2 + d4)>(%arg4, %arg2, %arg5, %arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)>(%arg4, %arg2, %arg5, %arg3, %arg6)
              %16 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %17 = affine.load %3[0, %arg4, 0, 0] : memref<1x8x1x1xf32>
              %18 = affine.load %alloc_16[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
              %19 = arith.addf %16, %17 : f32
              %20 = arith.maximumf %19, %cst : f32
              %21 = arith.maximumf %18, %20 : f32
              affine.store %21, %alloc_16[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
            }
          }
        }
      }
    }
  }
  %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 18 {
      affine.for %arg3 = 0 to 18 {
        affine.for %arg4 = 0 to 8 step 4 {
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %arg4] : memref<1x18x18x8xf32>
          %8 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg4)
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %8] : memref<1x18x18x8xf32>
          %9 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %9] : memref<1x18x18x8xf32>
          %10 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg4)
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %10] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %reinterpret_cast_18 = memref.reinterpret_cast %alloc_17 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_16, %reinterpret_cast_18 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 4 {
          affine.store %cst_12, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %9 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg3, %arg5)
              %10 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg6)
              affine.store %cst, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %11 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg3, %arg5)
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg6)
              affine.for %arg7 = 0 to 5 {
                affine.for %arg8 = 0 to 5 {
                  affine.for %arg9 = 0 to 8 step 4 {
                    %23 = affine.load %alloc_17[%c0_11, %11 + %arg7, %12 + %arg8, %arg9] : memref<1x18x18x8xf32>
                    %24 = affine.load %1[%arg2, %arg7, %arg8, %arg9] : memref<16x5x5x8xf32>
                    %25 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
                    %26 = arith.mulf %23, %24 : f32
                    %27 = arith.addf %25, %26 : f32
                    affine.store %27, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
                    %28 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg9)
                    %29 = affine.load %alloc_17[%c0_11, %11 + %arg7, %12 + %arg8, %28] : memref<1x18x18x8xf32>
                    %30 = affine.load %1[%arg2, %arg7, %arg8, %28] : memref<16x5x5x8xf32>
                    %31 = arith.mulf %29, %30 : f32
                    %32 = arith.addf %27, %31 : f32
                    affine.store %32, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
                    %33 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg9)
                    %34 = affine.load %alloc_17[%c0_11, %11 + %arg7, %12 + %arg8, %33] : memref<1x18x18x8xf32>
                    %35 = affine.load %1[%arg2, %arg7, %arg8, %33] : memref<16x5x5x8xf32>
                    %36 = arith.mulf %34, %35 : f32
                    %37 = arith.addf %32, %36 : f32
                    affine.store %37, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
                    %38 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg9)
                    %39 = affine.load %alloc_17[%c0_11, %11 + %arg7, %12 + %arg8, %38] : memref<1x18x18x8xf32>
                    %40 = affine.load %1[%arg2, %arg7, %arg8, %38] : memref<16x5x5x8xf32>
                    %41 = arith.mulf %39, %40 : f32
                    %42 = arith.addf %37, %41 : f32
                    affine.store %42, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
                  }
                }
              }
              %13 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (0)>(%arg2, %arg3, %arg5, %arg4, %arg6)
              %14 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1 * 3 + d2)>(%arg2, %arg3, %arg5, %arg4, %arg6)
              %15 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d3 * 3 + d4)>(%arg2, %arg3, %arg5, %arg4, %arg6)
              %16 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)>(%arg2, %arg3, %arg5, %arg4, %arg6)
              %17 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %18 = affine.load %2[0, %arg2, 0, 0] : memref<1x16x1x1xf32>
              %19 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %20 = arith.addf %17, %18 : f32
              %21 = arith.maximumf %20, %cst : f32
              %22 = arith.maximumf %19, %21 : f32
              affine.store %22, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
            }
          }
          %8 = affine.load %alloc[%arg1, 0, 0, 0] : memref<1x1x1x1xf32>
          affine.store %8, %alloc_19[%arg1, %arg2, %arg3, %arg4] : memref<1x16x4x4xf32>
        }
      }
    }
  }
  %reinterpret_cast_20 = memref.reinterpret_cast %alloc_19 to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<1x16x4x4xf32> to memref<1x1x256xf32>
  %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 10 {
        affine.store %cst, %alloc_21[%c0_8, %c0_7, %arg3] : memref<1x1x10xf32>
        affine.for %arg4 = 0 to 256 step 4 {
          %8 = affine.load %reinterpret_cast_20[%arg1, %arg2, %arg4] : memref<1x1x256xf32>
          %9 = affine.load %5[%arg1, %arg4, %arg3] : memref<1x256x10xf32>
          %10 = affine.load %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %11 = arith.mulf %8, %9 : f32
          %12 = arith.addf %10, %11 : f32
          affine.store %12, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %13 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg4)
          %14 = affine.load %reinterpret_cast_20[%arg1, %arg2, %13] : memref<1x1x256xf32>
          %15 = affine.load %5[%arg1, %13, %arg3] : memref<1x256x10xf32>
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %12, %16 : f32
          affine.store %17, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %18 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
          %19 = affine.load %reinterpret_cast_20[%arg1, %arg2, %18] : memref<1x1x256xf32>
          %20 = affine.load %5[%arg1, %18, %arg3] : memref<1x256x10xf32>
          %21 = arith.mulf %19, %20 : f32
          %22 = arith.addf %17, %21 : f32
          affine.store %22, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %23 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg4)
          %24 = affine.load %reinterpret_cast_20[%arg1, %arg2, %23] : memref<1x1x256xf32>
          %25 = affine.load %5[%arg1, %23, %arg3] : memref<1x256x10xf32>
          %26 = arith.mulf %24, %25 : f32
          %27 = arith.addf %22, %26 : f32
          affine.store %27, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
        }
      }
    }
  }
  %reinterpret_cast_22 = memref.reinterpret_cast %alloc_21 to offset: [0], sizes: [1, 10], strides: [10, 1] : memref<1x1x10xf32> to memref<1x10xf32>
  %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 8 step 4 {
      %8 = affine.load %reinterpret_cast_22[0, %arg2] : memref<1x10xf32>
      %9 = affine.load %6[0, %arg2] : memref<1x10xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %alloc_23[%arg1, %arg2] : memref<1x10xf32>
      %11 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg2)
      %12 = affine.load %reinterpret_cast_22[0, %11] : memref<1x10xf32>
      %13 = affine.load %6[0, %11] : memref<1x10xf32>
      %14 = arith.addf %12, %13 : f32
      affine.store %14, %alloc_23[%arg1, %11] : memref<1x10xf32>
      %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg2)
      %16 = affine.load %reinterpret_cast_22[0, %15] : memref<1x10xf32>
      %17 = affine.load %6[0, %15] : memref<1x10xf32>
      %18 = arith.addf %16, %17 : f32
      affine.store %18, %alloc_23[%arg1, %15] : memref<1x10xf32>
      %19 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg2)
      %20 = affine.load %reinterpret_cast_22[0, %19] : memref<1x10xf32>
      %21 = affine.load %6[0, %19] : memref<1x10xf32>
      %22 = arith.addf %20, %21 : f32
      affine.store %22, %alloc_23[%arg1, %19] : memref<1x10xf32>
    }
    affine.for %arg2 = 8 to 10 {
      %8 = affine.load %reinterpret_cast_22[0, %arg2] : memref<1x10xf32>
      %9 = affine.load %6[0, %arg2] : memref<1x10xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %alloc_23[%arg1, %arg2] : memref<1x10xf32>
    }
  }
  %7 = bufferization.to_tensor %alloc_23 : memref<1x10xf32>
  return %7 : tensor<1x10xf32>
}

