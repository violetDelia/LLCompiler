// -----// IR Dump After BufferLoopHoisting (buffer-loop-hoisting) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %2 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %3 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %4 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %5 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %6 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %0, %alloc : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2], [3]] : memref<1x1x28x28xf32> into memref<1x28x28xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 28, 28, 1] : memref<1x28x28xf32> into memref<1x28x28x1xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst, %alloc_1[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %subview = memref.subview %alloc_1[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 28 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst, %alloc_2[%arg1, %arg2, %arg3, %arg4] : memref<1x28x28x8xf32>
        }
      }
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 28 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 5 {
            affine.for %arg6 = 0 to 5 {
              affine.for %arg7 = 0 to 1 {
                %8 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %arg5)
                %9 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg3, %arg6)
                %10 = affine.load %alloc_1[%arg1, %8, %9, %arg7] : memref<1x32x32x1xf32>
                %11 = affine.load %4[%arg4, %arg5, %arg6, %arg7] : memref<8x5x5x1xf32>
                %12 = affine.load %alloc_2[%arg1, %arg2, %arg3, %arg4] : memref<1x28x28x8xf32>
                %13 = arith.mulf %10, %11 : f32
                %14 = arith.addf %12, %13 : f32
                affine.store %14, %alloc_2[%arg1, %arg2, %arg3, %arg4] : memref<1x28x28x8xf32>
              }
            }
          }
        }
      }
    }
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 14 {
      affine.for %arg3 = 0 to 14 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_0, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
        }
      }
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 14 {
      affine.for %arg3 = 0 to 14 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 2 {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg2, %arg5)
              %9 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              %10 = affine.load %alloc_2[%c0, %8, %9, %arg4] : memref<1x28x28x8xf32>
              %11 = affine.load %3[%c0, %arg4, %c0, %c0] : memref<1x8x1x1xf32>
              %12 = affine.load %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
              %13 = arith.addf %10, %11 : f32
              %14 = arith.maximumf %13, %cst : f32
              %15 = arith.maximumf %12, %14 : f32
              affine.store %15, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x8xf32>
            }
          }
        }
      }
    }
  }
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 18 {
      affine.for %arg3 = 0 to 18 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %subview_5 = memref.subview %alloc_4[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_3, %subview_5 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 14 {
      affine.for %arg3 = 0 to 14 {
        affine.for %arg4 = 0 to 16 {
          affine.store %cst, %alloc_6[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x16xf32>
        }
      }
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 14 {
      affine.for %arg3 = 0 to 14 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 5 {
            affine.for %arg6 = 0 to 5 {
              affine.for %arg7 = 0 to 8 {
                %8 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %arg5)
                %9 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg3, %arg6)
                %10 = affine.load %alloc_4[%arg1, %8, %9, %arg7] : memref<1x18x18x8xf32>
                %11 = affine.load %1[%arg4, %arg5, %arg6, %arg7] : memref<16x5x5x8xf32>
                %12 = affine.load %alloc_6[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x16xf32>
                %13 = arith.mulf %10, %11 : f32
                %14 = arith.addf %12, %13 : f32
                affine.store %14, %alloc_6[%arg1, %arg2, %arg3, %arg4] : memref<1x14x14x16xf32>
              }
            }
          }
        }
      }
    }
  }
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 16 {
          affine.store %cst_0, %alloc_7[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x16xf32>
        }
      }
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg2, %arg5)
              %9 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg3, %arg6)
              %10 = affine.load %alloc_6[%c0, %8, %9, %arg4] : memref<1x14x14x16xf32>
              %11 = affine.load %2[%c0, %arg4, %c0, %c0] : memref<1x16x1x1xf32>
              %12 = affine.load %alloc_7[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x16xf32>
              %13 = arith.addf %10, %11 : f32
              %14 = arith.maximumf %13, %cst : f32
              %15 = arith.maximumf %12, %14 : f32
              affine.store %15, %alloc_7[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x16xf32>
            }
          }
        }
      }
    }
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 4 {
          %8 = affine.load %alloc_7[%arg1, %arg3, %arg4, %arg2] : memref<1x4x4x16xf32>
          affine.store %8, %alloc_8[%arg1, %arg2, %arg3, %arg4] : memref<1x16x4x4xf32>
        }
      }
    }
  }
  %collapse_shape_9 = memref.collapse_shape %alloc_8 [[0], [1, 2, 3]] : memref<1x16x4x4xf32> into memref<1x256xf32>
  %expand_shape_10 = memref.expand_shape %collapse_shape_9 [[0, 1], [2]] output_shape [1, 1, 256] : memref<1x256xf32> into memref<1x1x256xf32>
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 10 {
        affine.store %cst, %alloc_11[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
      }
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 10 {
        affine.for %arg4 = 0 to 256 {
          %8 = affine.load %expand_shape_10[%arg1, %arg2, %arg4] : memref<1x1x256xf32>
          %9 = affine.load %5[%arg1, %arg4, %arg3] : memref<1x256x10xf32>
          %10 = affine.load %alloc_11[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %11 = arith.mulf %8, %9 : f32
          %12 = arith.addf %10, %11 : f32
          affine.store %12, %alloc_11[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
        }
      }
    }
  }
  %collapse_shape_12 = memref.collapse_shape %alloc_11 [[0, 1], [2]] : memref<1x1x10xf32> into memref<1x10xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 10 {
      %8 = affine.load %collapse_shape_12[%c0, %arg2] : memref<1x10xf32>
      %9 = affine.load %6[%c0, %arg2] : memref<1x10xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %alloc_13[%arg1, %arg2] : memref<1x10xf32>
    }
  }
  %7 = bufferization.to_tensor %alloc_13 : memref<1x10xf32>
  return %7 : tensor<1x10xf32>
}

