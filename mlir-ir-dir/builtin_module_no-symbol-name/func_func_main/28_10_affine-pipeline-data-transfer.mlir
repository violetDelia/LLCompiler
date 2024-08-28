// -----// IR Dump After AffinePipelineDataTransfer (affine-pipeline-data-transfer) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x10xf32> {
  %cst = arith.constant -3.40282347E+38 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %alloc_1 = memref.alloc() : memref<1x1x1x1xf32>
  %alloc_2 = memref.alloc() : memref<1x1x1x1xf32>
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc_3 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %collapse_shape = memref.collapse_shape %alloc_3 [[0, 1], [2], [3]] : memref<1x1x28x28xf32> into memref<1x28x28xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 28, 28, 1] : memref<1x28x28xf32> into memref<1x28x28x1xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_0, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %subview = memref.subview %alloc_4[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 14 {
      affine.for %arg3 = 0 to 14 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst, %alloc_5[%c0, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 2 {
              affine.store %cst_0, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %6 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg2, %arg5)
              %7 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              affine.for %arg7 = 0 to 5 {
                affine.for %arg8 = 0 to 5 {
                  %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%6, %arg7)
                  %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%7, %arg8)
                  %16 = affine.load %alloc_4[%c0, %14, %15, %c0] : memref<1x32x32x1xf32>
                  %17 = affine.load %3[%arg4, %arg7, %arg8, %c0] : memref<8x5x5x1xf32>
                  %18 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                  %19 = arith.mulf %16, %17 : f32
                  %20 = arith.addf %18, %19 : f32
                  affine.store %20, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
                }
              }
              %8 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %9 = affine.load %2[%c0, %arg4, %c0, %c0] : memref<1x8x1x1xf32>
              %10 = affine.load %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
              %11 = arith.addf %8, %9 : f32
              %12 = arith.maximumf %11, %cst_0 : f32
              %13 = arith.maximumf %10, %12 : f32
              affine.store %13, %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
            }
          }
        }
      }
    }
  }
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 18 {
      affine.for %arg3 = 0 to 18 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_0, %alloc_6[%arg1, %arg2, %arg3, %arg4] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %subview_7 = memref.subview %alloc_6[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_5, %subview_7 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 4 {
          affine.store %cst, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              affine.store %cst_0, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %7 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg3, %arg5)
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg6)
              affine.for %arg7 = 0 to 5 {
                affine.for %arg8 = 0 to 5 {
                  affine.for %arg9 = 0 to 8 {
                    %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%7, %arg7)
                    %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%8, %arg8)
                    %17 = affine.load %alloc_6[%c0, %15, %16, %arg9] : memref<1x18x18x8xf32>
                    %18 = affine.load %0[%arg2, %arg7, %arg8, %arg9] : memref<16x5x5x8xf32>
                    %19 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
                    %20 = arith.mulf %17, %18 : f32
                    %21 = arith.addf %19, %20 : f32
                    affine.store %21, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
                  }
                }
              }
              %9 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %10 = affine.load %1[%c0, %arg2, %c0, %c0] : memref<1x16x1x1xf32>
              %11 = affine.load %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %12 = arith.addf %9, %10 : f32
              %13 = arith.maximumf %12, %cst_0 : f32
              %14 = arith.maximumf %11, %13 : f32
              affine.store %14, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
            }
          }
          %6 = affine.load %alloc_1[%arg1, 0, 0, 0] : memref<1x1x1x1xf32>
          affine.store %6, %alloc_8[%arg1, %arg2, %arg3, %arg4] : memref<1x16x4x4xf32>
        }
      }
    }
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 10 {
        affine.store %cst_0, %alloc_9[%c0, %c0, %arg3] : memref<1x1x10xf32>
        affine.for %arg4 = 0 to 256 {
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg1, %arg2]
          %7 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%arg4)
          %8 = affine.apply affine_map<(d0) -> ((d0 mod 16) floordiv 4)>(%arg4)
          %9 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg4)
          %10 = affine.load %alloc_8[%6, %7, %8, %9] : memref<1x16x4x4xf32>
          %11 = affine.load %4[%arg1, %arg4, %arg3] : memref<1x256x10xf32>
          %12 = affine.load %alloc_9[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %13 = arith.mulf %10, %11 : f32
          %14 = arith.addf %12, %13 : f32
          affine.store %14, %alloc_9[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
        }
      }
    }
  }
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 10 {
      %6 = affine.load %alloc_9[%c0, %c0, %arg2] : memref<1x1x10xf32>
      %7 = affine.load %5[%c0, %arg2] : memref<1x10xf32>
      %8 = arith.addf %6, %7 : f32
      affine.store %8, %alloc_10[%arg1, %arg2] : memref<1x10xf32>
    }
  }
  return %alloc_10 : memref<1x10xf32>
}

