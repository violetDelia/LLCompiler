// -----// IR Dump After LoopCoalescing (affine-loop-coalescing) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x10xf32>) {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %alloc_3 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_4 = arith.constant 0 : index
  %c0_5 = arith.constant 0 : index
  %alloc_6 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_7 = arith.constant 0 : index
  %c0_8 = arith.constant 0 : index
  %c0_9 = arith.constant 0 : index
  %c0_10 = arith.constant 0 : index
  %c0_11 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_12 = arith.constant -3.40282347E+38 : f32
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc_13 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1], [2], [3]] : memref<1x1x28x28xf32> into memref<1x28x28xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 28, 28, 1] : memref<1x28x28xf32> into memref<1x28x28x1xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %6 = affine.apply affine_map<() -> (1)>()
  %7 = affine.apply affine_map<() -> (32)>()
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%6)[%7]
  %9 = affine.apply affine_map<() -> (32)>()
  %10 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%8)[%9]
  %11 = affine.apply affine_map<() -> (1)>()
  %12 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%10)[%11]
  affine.parallel (%arg2) = (0) to (symbol(%12)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%11]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%11]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%9]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%9]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%7]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%7]
    affine.store %cst, %alloc_14[%47, %46, %44, %42] : memref<1x32x32x1xf32>
  }
  %subview = memref.subview %alloc_14[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %13 = affine.apply affine_map<() -> (1)>()
  %14 = affine.apply affine_map<() -> (14)>()
  %15 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%13)[%14]
  %16 = affine.apply affine_map<() -> (14)>()
  %17 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%15)[%16]
  %18 = affine.apply affine_map<() -> (8)>()
  %19 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%17)[%18]
  affine.for %arg2 = 0 to %19 {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%18]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%18]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%16]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%16]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%14]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%14]
    affine.store %cst_12, %alloc_15[%c0_10, %46, %44, %42] : memref<1x14x14x8xf32>
    %48 = affine.apply affine_map<() -> (2)>()
    %49 = affine.apply affine_map<() -> (2)>()
    %50 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%48)[%49]
    affine.for %arg3 = 0 to %50 {
      %51 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%49]
      %52 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%49]
      %53 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%46, %52)
      %54 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%44, %51)
      affine.store %cst, %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %55 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%46, %52)
      %56 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%44, %51)
      %57 = affine.apply affine_map<() -> (5)>()
      %58 = affine.apply affine_map<() -> (5)>()
      %59 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%57)[%58]
      affine.for %arg4 = 0 to %59 {
        %68 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg4)[%58]
        %69 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg4)[%58]
        %70 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%55, %69)
        %71 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%56, %68)
        %72 = affine.load %alloc_14[%c0_9, %70, %71, %c0_8] : memref<1x32x32x1xf32>
        %73 = affine.load %3[%42, %69, %68, %c0_8] : memref<8x5x5x1xf32>
        %74 = affine.load %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %75 = arith.mulf %72, %73 : f32
        %76 = arith.addf %74, %75 : f32
        affine.store %76, %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %60 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%46, %52)
      %61 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%44, %51)
      %62 = affine.load %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %63 = affine.load %2[%c0_11, %42, %c0_11, %c0_11] : memref<1x8x1x1xf32>
      %64 = affine.load %alloc_15[%47, %46, %44, %42] : memref<1x14x14x8xf32>
      %65 = arith.addf %62, %63 : f32
      %66 = arith.maximumf %65, %cst : f32
      %67 = arith.maximumf %64, %66 : f32
      affine.store %67, %alloc_15[%47, %46, %44, %42] : memref<1x14x14x8xf32>
    }
  }
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %20 = affine.apply affine_map<() -> (1)>()
  %21 = affine.apply affine_map<() -> (18)>()
  %22 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%20)[%21]
  %23 = affine.apply affine_map<() -> (18)>()
  %24 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%22)[%23]
  %25 = affine.apply affine_map<() -> (8)>()
  %26 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%24)[%25]
  affine.parallel (%arg2) = (0) to (symbol(%26)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%25]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%25]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%23]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%23]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%21]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%21]
    affine.store %cst, %alloc_16[%47, %46, %44, %42] : memref<1x18x18x8xf32>
  }
  %subview_17 = memref.subview %alloc_16[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_15, %subview_17 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %27 = affine.apply affine_map<() -> (1)>()
  %28 = affine.apply affine_map<() -> (16)>()
  %29 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%27)[%28]
  %30 = affine.apply affine_map<() -> (4)>()
  %31 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%29)[%30]
  %32 = affine.apply affine_map<() -> (4)>()
  %33 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%31)[%32]
  affine.for %arg2 = 0 to %33 {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%32]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%32]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%30]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%30]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%28]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%28]
    affine.store %cst_12, %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
    %48 = affine.apply affine_map<() -> (3)>()
    %49 = affine.apply affine_map<() -> (3)>()
    %50 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%48)[%49]
    affine.for %arg3 = 0 to %50 {
      %52 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%49]
      %53 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%49]
      %54 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%44, %53)
      %55 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%42, %52)
      affine.store %cst, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %56 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%44, %53)
      %57 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%42, %52)
      %58 = affine.apply affine_map<() -> (5)>()
      %59 = affine.apply affine_map<() -> (5)>()
      %60 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%58)[%59]
      %61 = affine.apply affine_map<() -> (8)>()
      %62 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%60)[%61]
      affine.for %arg4 = 0 to %62 {
        %71 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg4)[%61]
        %72 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg4)[%61]
        %73 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%72)[%59]
        %74 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%72)[%59]
        %75 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%56, %74)
        %76 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%57, %73)
        %77 = affine.load %alloc_16[%c0_2, %75, %76, %71] : memref<1x18x18x8xf32>
        %78 = affine.load %0[%46, %74, %73, %71] : memref<16x5x5x8xf32>
        %79 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %80 = arith.mulf %77, %78 : f32
        %81 = arith.addf %79, %80 : f32
        affine.store %81, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %63 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%44, %53)
      %64 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%42, %52)
      %65 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %66 = affine.load %1[%c0_11, %46, %c0_11, %c0_11] : memref<1x16x1x1xf32>
      %67 = affine.load %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %68 = arith.addf %65, %66 : f32
      %69 = arith.maximumf %68, %cst : f32
      %70 = arith.maximumf %67, %69 : f32
      affine.store %70, %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
    }
    %51 = affine.load %alloc_3[%47, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.store %51, %alloc_18[%47, %46, %44, %42] : memref<1x16x4x4xf32>
  }
  %collapse_shape_19 = memref.collapse_shape %alloc_18 [[0], [1, 2, 3]] : memref<1x16x4x4xf32> into memref<1x256xf32>
  %expand_shape_20 = memref.expand_shape %collapse_shape_19 [[0, 1], [2]] output_shape [1, 1, 256] : memref<1x256xf32> into memref<1x1x256xf32>
  %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %34 = affine.apply affine_map<() -> (1)>()
  %35 = affine.apply affine_map<() -> (1)>()
  %36 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%34)[%35]
  %37 = affine.apply affine_map<() -> (10)>()
  %38 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%36)[%37]
  affine.parallel (%arg2) = (0) to (symbol(%38)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%37]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%37]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%35]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%35]
    affine.store %cst, %alloc_21[%c0_0, %c0, %42] : memref<1x1x10xf32>
    affine.for %arg3 = 0 to 256 {
      %46 = affine.load %expand_shape_20[%45, %44, %arg3] : memref<1x1x256xf32>
      %47 = affine.load %4[%45, %arg3, %42] : memref<1x256x10xf32>
      %48 = affine.load %alloc_21[%45, %44, %42] : memref<1x1x10xf32>
      %49 = arith.mulf %46, %47 : f32
      %50 = arith.addf %48, %49 : f32
      affine.store %50, %alloc_21[%45, %44, %42] : memref<1x1x10xf32>
    }
  }
  %collapse_shape_22 = memref.collapse_shape %alloc_21 [[0, 1], [2]] : memref<1x1x10xf32> into memref<1x10xf32>
  %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %39 = affine.apply affine_map<() -> (1)>()
  %40 = affine.apply affine_map<() -> (10)>()
  %41 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%39)[%40]
  affine.parallel (%arg2) = (0) to (symbol(%41)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%40]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%40]
    %44 = affine.load %collapse_shape_22[%c0_11, %42] : memref<1x10xf32>
    %45 = affine.load %5[%c0_11, %42] : memref<1x10xf32>
    %46 = arith.addf %44, %45 : f32
    affine.store %46, %alloc_23[%43, %42] : memref<1x10xf32>
  }
  memref.copy %alloc_23, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  return
}

