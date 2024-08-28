// -----// IR Dump After LoopCoalescing (affine-loop-coalescing) //----- //
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
  %6 = affine.apply affine_map<() -> (1)>()
  %7 = affine.apply affine_map<() -> (32)>()
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%6)[%7]
  %9 = affine.apply affine_map<() -> (32)>()
  %10 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%8)[%9]
  %11 = affine.apply affine_map<() -> (1)>()
  %12 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%10)[%11]
  affine.parallel (%arg1) = (0) to (symbol(%12)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%11]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%11]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%9]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%9]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%7]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%7]
    affine.store %cst_0, %alloc_4[%47, %46, %44, %42] : memref<1x32x32x1xf32>
  }
  %subview = memref.subview %alloc_4[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %13 = affine.apply affine_map<() -> (1)>()
  %14 = affine.apply affine_map<() -> (14)>()
  %15 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%13)[%14]
  %16 = affine.apply affine_map<() -> (14)>()
  %17 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%15)[%16]
  %18 = affine.apply affine_map<() -> (8)>()
  %19 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%17)[%18]
  affine.for %arg1 = 0 to %19 {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%18]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%18]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%16]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%16]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%14]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%14]
    affine.store %cst, %alloc_5[%c0, %46, %44, %42] : memref<1x14x14x8xf32>
    %48 = affine.apply affine_map<() -> (2)>()
    %49 = affine.apply affine_map<() -> (2)>()
    %50 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%48)[%49]
    affine.for %arg2 = 0 to %50 {
      %51 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%49]
      %52 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%49]
      affine.store %cst_0, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %53 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%46, %52)
      %54 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%44, %51)
      %55 = affine.apply affine_map<() -> (5)>()
      %56 = affine.apply affine_map<() -> (5)>()
      %57 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%55)[%56]
      affine.for %arg3 = 0 to %57 {
        %64 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%56]
        %65 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%56]
        %66 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%53, %65)
        %67 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%54, %64)
        %68 = affine.load %alloc_4[%c0, %66, %67, %c0] : memref<1x32x32x1xf32>
        %69 = affine.load %3[%42, %65, %64, %c0] : memref<8x5x5x1xf32>
        %70 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %71 = arith.mulf %68, %69 : f32
        %72 = arith.addf %70, %71 : f32
        affine.store %72, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %58 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %59 = affine.load %2[%c0, %42, %c0, %c0] : memref<1x8x1x1xf32>
      %60 = affine.load %alloc_5[%47, %46, %44, %42] : memref<1x14x14x8xf32>
      %61 = arith.addf %58, %59 : f32
      %62 = arith.maximumf %61, %cst_0 : f32
      %63 = arith.maximumf %60, %62 : f32
      affine.store %63, %alloc_5[%47, %46, %44, %42] : memref<1x14x14x8xf32>
    }
  }
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %20 = affine.apply affine_map<() -> (1)>()
  %21 = affine.apply affine_map<() -> (18)>()
  %22 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%20)[%21]
  %23 = affine.apply affine_map<() -> (18)>()
  %24 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%22)[%23]
  %25 = affine.apply affine_map<() -> (8)>()
  %26 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%24)[%25]
  affine.parallel (%arg1) = (0) to (symbol(%26)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%25]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%25]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%23]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%23]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%21]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%21]
    affine.store %cst_0, %alloc_6[%47, %46, %44, %42] : memref<1x18x18x8xf32>
  }
  %subview_7 = memref.subview %alloc_6[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_5, %subview_7 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %27 = affine.apply affine_map<() -> (1)>()
  %28 = affine.apply affine_map<() -> (16)>()
  %29 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%27)[%28]
  %30 = affine.apply affine_map<() -> (4)>()
  %31 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%29)[%30]
  %32 = affine.apply affine_map<() -> (4)>()
  %33 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%31)[%32]
  affine.for %arg1 = 0 to %33 {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%32]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%32]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%30]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%30]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%28]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%28]
    affine.store %cst, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
    %48 = affine.apply affine_map<() -> (3)>()
    %49 = affine.apply affine_map<() -> (3)>()
    %50 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%48)[%49]
    affine.for %arg2 = 0 to %50 {
      %52 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%49]
      %53 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%49]
      affine.store %cst_0, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %54 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%44, %53)
      %55 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%42, %52)
      %56 = affine.apply affine_map<() -> (5)>()
      %57 = affine.apply affine_map<() -> (5)>()
      %58 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%56)[%57]
      %59 = affine.apply affine_map<() -> (8)>()
      %60 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%58)[%59]
      affine.for %arg3 = 0 to %60 {
        %67 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%59]
        %68 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%59]
        %69 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%68)[%57]
        %70 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%68)[%57]
        %71 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%54, %70)
        %72 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%55, %69)
        %73 = affine.load %alloc_6[%c0, %71, %72, %67] : memref<1x18x18x8xf32>
        %74 = affine.load %0[%46, %70, %69, %67] : memref<16x5x5x8xf32>
        %75 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %76 = arith.mulf %73, %74 : f32
        %77 = arith.addf %75, %76 : f32
        affine.store %77, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %61 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %62 = affine.load %1[%c0, %46, %c0, %c0] : memref<1x16x1x1xf32>
      %63 = affine.load %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %64 = arith.addf %61, %62 : f32
      %65 = arith.maximumf %64, %cst_0 : f32
      %66 = arith.maximumf %63, %65 : f32
      affine.store %66, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
    }
    %51 = affine.load %alloc_1[%47, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.store %51, %alloc_8[%47, %46, %44, %42] : memref<1x16x4x4xf32>
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %34 = affine.apply affine_map<() -> (1)>()
  %35 = affine.apply affine_map<() -> (1)>()
  %36 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%34)[%35]
  %37 = affine.apply affine_map<() -> (10)>()
  %38 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%36)[%37]
  affine.parallel (%arg1) = (0) to (symbol(%38)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%37]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%37]
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%43)[%35]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%43)[%35]
    affine.store %cst_0, %alloc_9[%c0, %c0, %42] : memref<1x1x10xf32>
    affine.for %arg2 = 0 to 256 {
      %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%45, %44]
      %47 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%arg2)
      %48 = affine.apply affine_map<(d0) -> ((d0 mod 16) floordiv 4)>(%arg2)
      %49 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg2)
      %50 = affine.load %alloc_8[%46, %47, %48, %49] : memref<1x16x4x4xf32>
      %51 = affine.load %4[%45, %arg2, %42] : memref<1x256x10xf32>
      %52 = affine.load %alloc_9[%45, %44, %42] : memref<1x1x10xf32>
      %53 = arith.mulf %50, %51 : f32
      %54 = arith.addf %52, %53 : f32
      affine.store %54, %alloc_9[%45, %44, %42] : memref<1x1x10xf32>
    }
  }
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %39 = affine.apply affine_map<() -> (1)>()
  %40 = affine.apply affine_map<() -> (10)>()
  %41 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%39)[%40]
  affine.parallel (%arg1) = (0) to (symbol(%41)) {
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%40]
    %43 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%40]
    %44 = affine.load %alloc_9[%c0, %c0, %42] : memref<1x1x10xf32>
    %45 = affine.load %5[%c0, %42] : memref<1x10xf32>
    %46 = arith.addf %44, %45 : f32
    affine.store %46, %alloc_10[%43, %42] : memref<1x10xf32>
  }
  return %alloc_10 : memref<1x10xf32>
}

