// -----// IR Dump After LoopCoalescing (affine-loop-coalescing) //----- //
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
  %6 = affine.apply affine_map<() -> (1)>()
  %7 = affine.apply affine_map<() -> (32)>()
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%6)[%7]
  %9 = affine.apply affine_map<() -> (32)>()
  %10 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%8)[%9]
  %11 = affine.apply affine_map<() -> (1)>()
  %12 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%10)[%11]
  affine.for %arg2 = 0 to %12 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%11]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%11]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%9]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%9]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%7]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%7]
    affine.store %cst, %alloc_6[%77, %76, %74, %72] : memref<1x32x32x1xf32>
  }
  %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_7 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc : memref<1x1x28x28xf32>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  %13 = affine.apply affine_map<() -> (1)>()
  %14 = affine.apply affine_map<() -> (28)>()
  %15 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%13)[%14]
  %16 = affine.apply affine_map<() -> (28)>()
  %17 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%15)[%16]
  %18 = affine.apply affine_map<() -> (8)>()
  %19 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%17)[%18]
  affine.for %arg2 = 0 to %19 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%18]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%18]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%16]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%16]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%14]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%14]
    affine.store %cst, %alloc_8[%77, %76, %74, %72] : memref<1x28x28x8xf32>
  }
  %20 = affine.apply affine_map<() -> (1)>()
  %21 = affine.apply affine_map<() -> (28)>()
  %22 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%20)[%21]
  %23 = affine.apply affine_map<() -> (28)>()
  %24 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%22)[%23]
  %25 = affine.apply affine_map<() -> (8)>()
  %26 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%24)[%25]
  %27 = affine.apply affine_map<() -> (1)>()
  %28 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%26)[%27]
  affine.for %arg2 = 0 to %28 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%27]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%27]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%25]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%25]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%23]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%23]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%21]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%21]
    affine.store %cst, %alloc_8[%c0_2, %78, %76, %74] : memref<1x28x28x8xf32>
    %80 = affine.apply affine_map<() -> (5)>()
    %81 = affine.apply affine_map<() -> (5)>()
    %82 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%80)[%81]
    affine.for %arg3 = 0 to %82 {
      %83 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%81]
      %84 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%81]
      %85 = affine.load %alloc_6[%79, %78 + %84, %76 + %83, %72] : memref<1x32x32x1xf32>
      %86 = affine.load %3[%74, %84, %83, %72] : memref<8x5x5x1xf32>
      %87 = affine.load %alloc_8[%79, %78, %76, %74] : memref<1x28x28x8xf32>
      %88 = arith.mulf %85, %86 : f32
      %89 = arith.addf %87, %88 : f32
      affine.store %89, %alloc_8[%79, %78, %76, %74] : memref<1x28x28x8xf32>
    }
  }
  memref.dealloc %alloc_6 : memref<1x32x32x1xf32>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %29 = affine.apply affine_map<() -> (1)>()
  %30 = affine.apply affine_map<() -> (14)>()
  %31 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%29)[%30]
  %32 = affine.apply affine_map<() -> (14)>()
  %33 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%31)[%32]
  %34 = affine.apply affine_map<() -> (8)>()
  %35 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%33)[%34]
  affine.for %arg2 = 0 to %35 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%34]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%34]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%32]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%32]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%30]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%30]
    affine.store %cst_5, %alloc_9[%c0_4, %76, %74, %72] : memref<1x14x14x8xf32>
    %78 = affine.apply affine_map<() -> (2)>()
    %79 = affine.apply affine_map<() -> (2)>()
    %80 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%78)[%79]
    affine.for %arg3 = 0 to %80 {
      %81 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%79]
      %82 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%79]
      %83 = affine.load %alloc_8[0, %76 * 2 + %82, %74 * 2 + %81, %72] : memref<1x28x28x8xf32>
      %84 = affine.load %2[0, %72, 0, 0] : memref<1x8x1x1xf32>
      %85 = affine.load %alloc_9[%77, %76, %74, %72] : memref<1x14x14x8xf32>
      %86 = arith.addf %83, %84 : f32
      %87 = arith.maximumf %86, %cst : f32
      %88 = arith.maximumf %85, %87 : f32
      affine.store %88, %alloc_9[%77, %76, %74, %72] : memref<1x14x14x8xf32>
    }
  }
  memref.dealloc %alloc_8 : memref<1x28x28x8xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %36 = affine.apply affine_map<() -> (1)>()
  %37 = affine.apply affine_map<() -> (18)>()
  %38 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%36)[%37]
  %39 = affine.apply affine_map<() -> (18)>()
  %40 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%38)[%39]
  %41 = affine.apply affine_map<() -> (8)>()
  %42 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%40)[%41]
  affine.for %arg2 = 0 to %42 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%41]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%41]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%39]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%39]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%37]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%37]
    affine.store %cst, %alloc_10[%77, %76, %74, %72] : memref<1x18x18x8xf32>
  }
  %reinterpret_cast_11 = memref.reinterpret_cast %alloc_10 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_9, %reinterpret_cast_11 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_9 : memref<1x14x14x8xf32>
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  %43 = affine.apply affine_map<() -> (1)>()
  %44 = affine.apply affine_map<() -> (14)>()
  %45 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%43)[%44]
  %46 = affine.apply affine_map<() -> (14)>()
  %47 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%45)[%46]
  %48 = affine.apply affine_map<() -> (16)>()
  %49 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%47)[%48]
  affine.for %arg2 = 0 to %49 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%48]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%48]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%46]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%46]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%44]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%44]
    affine.store %cst, %alloc_12[%c0_3, %76, %74, %72] : memref<1x14x14x16xf32>
    %78 = affine.apply affine_map<() -> (5)>()
    %79 = affine.apply affine_map<() -> (5)>()
    %80 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%78)[%79]
    %81 = affine.apply affine_map<() -> (8)>()
    %82 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%80)[%81]
    affine.for %arg3 = 0 to %82 {
      %83 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%81]
      %84 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%81]
      %85 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%84)[%79]
      %86 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%84)[%79]
      %87 = affine.load %alloc_10[%77, %76 + %86, %74 + %85, %83] : memref<1x18x18x8xf32>
      %88 = affine.load %0[%72, %86, %85, %83] : memref<16x5x5x8xf32>
      %89 = affine.load %alloc_12[%77, %76, %74, %72] : memref<1x14x14x16xf32>
      %90 = arith.mulf %87, %88 : f32
      %91 = arith.addf %89, %90 : f32
      affine.store %91, %alloc_12[%77, %76, %74, %72] : memref<1x14x14x16xf32>
    }
  }
  memref.dealloc %alloc_10 : memref<1x18x18x8xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  %50 = affine.apply affine_map<() -> (1)>()
  %51 = affine.apply affine_map<() -> (4)>()
  %52 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%50)[%51]
  %53 = affine.apply affine_map<() -> (4)>()
  %54 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%52)[%53]
  %55 = affine.apply affine_map<() -> (16)>()
  %56 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%54)[%55]
  affine.for %arg2 = 0 to %56 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%55]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%55]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%53]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%53]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%51]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%51]
    affine.store %cst_5, %alloc_13[%c0, %76, %74, %72] : memref<1x4x4x16xf32>
    %78 = affine.apply affine_map<() -> (3)>()
    %79 = affine.apply affine_map<() -> (3)>()
    %80 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%78)[%79]
    affine.for %arg3 = 0 to %80 {
      %81 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%79]
      %82 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%79]
      %83 = affine.load %alloc_12[0, %76 * 3 + %82, %74 * 3 + %81, %72] : memref<1x14x14x16xf32>
      %84 = affine.load %1[0, %72, 0, 0] : memref<1x16x1x1xf32>
      %85 = affine.load %alloc_13[%77, %76, %74, %72] : memref<1x4x4x16xf32>
      %86 = arith.addf %83, %84 : f32
      %87 = arith.maximumf %86, %cst : f32
      %88 = arith.maximumf %85, %87 : f32
      affine.store %88, %alloc_13[%77, %76, %74, %72] : memref<1x4x4x16xf32>
    }
  }
  memref.dealloc %alloc_12 : memref<1x14x14x16xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %57 = affine.apply affine_map<() -> (1)>()
  %58 = affine.apply affine_map<() -> (16)>()
  %59 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%57)[%58]
  %60 = affine.apply affine_map<() -> (4)>()
  %61 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%59)[%60]
  %62 = affine.apply affine_map<() -> (4)>()
  %63 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%61)[%62]
  affine.for %arg2 = 0 to %63 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%62]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%62]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%60]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%60]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%58]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%58]
    %78 = affine.load %alloc_13[%77, %74, %72, %76] : memref<1x4x4x16xf32>
    affine.store %78, %alloc_14[%77, %76, %74, %72] : memref<1x16x4x4xf32>
  }
  memref.dealloc %alloc_13 : memref<1x4x4x16xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %64 = affine.apply affine_map<() -> (1)>()
  %65 = affine.apply affine_map<() -> (1)>()
  %66 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%64)[%65]
  %67 = affine.apply affine_map<() -> (10)>()
  %68 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%66)[%67]
  affine.for %arg2 = 0 to %68 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%67]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%67]
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%73)[%65]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%73)[%65]
    affine.store %cst, %alloc_15[%c0_1, %c0_0, %72] : memref<1x1x10xf32>
    affine.for %arg3 = 0 to 256 {
      %76 = affine.load %alloc_14[symbol(%75) + symbol(%74), %arg3 floordiv 16, (%arg3 mod 16) floordiv 4, %arg3 mod 4] : memref<1x16x4x4xf32>
      %77 = affine.load %4[%75, %arg3, %72] : memref<1x256x10xf32>
      %78 = affine.load %alloc_15[%75, %74, %72] : memref<1x1x10xf32>
      %79 = arith.mulf %76, %77 : f32
      %80 = arith.addf %78, %79 : f32
      affine.store %80, %alloc_15[%75, %74, %72] : memref<1x1x10xf32>
    }
  }
  memref.dealloc %alloc_14 : memref<1x16x4x4xf32>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %69 = affine.apply affine_map<() -> (1)>()
  %70 = affine.apply affine_map<() -> (10)>()
  %71 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%69)[%70]
  affine.for %arg2 = 0 to %71 {
    %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%70]
    %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%70]
    %74 = affine.load %alloc_15[0, 0, %72] : memref<1x1x10xf32>
    %75 = affine.load %5[0, %72] : memref<1x10xf32>
    %76 = arith.addf %74, %75 : f32
    affine.store %76, %alloc_16[%73, %72] : memref<1x10xf32>
  }
  memref.dealloc %alloc_15 : memref<1x1x10xf32>
  memref.copy %alloc_16, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  memref.dealloc %alloc_16 : memref<1x10xf32>
  return
}

