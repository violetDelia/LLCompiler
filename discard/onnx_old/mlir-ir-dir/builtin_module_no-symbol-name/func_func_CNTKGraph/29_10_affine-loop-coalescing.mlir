// -----// IR Dump After LoopCoalescing (affine-loop-coalescing) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
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
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %7 = affine.apply affine_map<() -> (1)>()
  %8 = affine.apply affine_map<() -> (32)>()
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%7)[%8]
  %10 = affine.apply affine_map<() -> (32)>()
  %11 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%9)[%10]
  %12 = affine.apply affine_map<() -> (1)>()
  %13 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%11)[%12]
  affine.for %arg1 = 0 to %13 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%12]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%12]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%10]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%10]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%8]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%8]
    affine.store %cst, %alloc_1[%125, %124, %122, %120] : memref<1x32x32x1xf32>
  }
  %reinterpret_cast_2 = memref.reinterpret_cast %alloc_1 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_2 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  %14 = affine.apply affine_map<() -> (1)>()
  %15 = affine.apply affine_map<() -> (28)>()
  %16 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%14)[%15]
  %17 = affine.apply affine_map<() -> (28)>()
  %18 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%16)[%17]
  %19 = affine.apply affine_map<() -> (8)>()
  %20 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%18)[%19]
  affine.for %arg1 = 0 to %20 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%19]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%19]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%17]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%17]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%15]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%15]
    affine.store %cst, %alloc_3[%125, %124, %122, %120] : memref<1x28x28x8xf32>
  }
  %21 = affine.apply affine_map<() -> (1)>()
  %22 = affine.apply affine_map<() -> (28)>()
  %23 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%21)[%22]
  %24 = affine.apply affine_map<() -> (28)>()
  %25 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%23)[%24]
  %26 = affine.apply affine_map<() -> (8)>()
  %27 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%25)[%26]
  %28 = affine.apply affine_map<() -> (5)>()
  %29 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%27)[%28]
  %30 = affine.apply affine_map<() -> (5)>()
  %31 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%29)[%30]
  %32 = affine.apply affine_map<() -> (1)>()
  %33 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%31)[%32]
  affine.for %arg1 = 0 to %33 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%32]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%32]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%30]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%30]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%28]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%28]
    %126 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%125)[%26]
    %127 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%125)[%26]
    %128 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%127)[%24]
    %129 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%127)[%24]
    %130 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%129)[%22]
    %131 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%129)[%22]
    %132 = affine.load %alloc_1[%131, %130 + %124, %128 + %122, %120] : memref<1x32x32x1xf32>
    %133 = affine.load %4[%126, %124, %122, %120] : memref<8x5x5x1xf32>
    %134 = affine.load %alloc_3[%131, %130, %128, %126] : memref<1x28x28x8xf32>
    %135 = arith.mulf %132, %133 : f32
    %136 = arith.addf %134, %135 : f32
    affine.store %136, %alloc_3[%131, %130, %128, %126] : memref<1x28x28x8xf32>
  }
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %34 = affine.apply affine_map<() -> (1)>()
  %35 = affine.apply affine_map<() -> (14)>()
  %36 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%34)[%35]
  %37 = affine.apply affine_map<() -> (14)>()
  %38 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%36)[%37]
  %39 = affine.apply affine_map<() -> (8)>()
  %40 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%38)[%39]
  affine.for %arg1 = 0 to %40 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%39]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%39]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%37]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%37]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%35]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%35]
    affine.store %cst_0, %alloc_4[%125, %124, %122, %120] : memref<1x14x14x8xf32>
  }
  %41 = affine.apply affine_map<() -> (1)>()
  %42 = affine.apply affine_map<() -> (14)>()
  %43 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%41)[%42]
  %44 = affine.apply affine_map<() -> (14)>()
  %45 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%43)[%44]
  %46 = affine.apply affine_map<() -> (8)>()
  %47 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%45)[%46]
  %48 = affine.apply affine_map<() -> (2)>()
  %49 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%47)[%48]
  %50 = affine.apply affine_map<() -> (2)>()
  %51 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%49)[%50]
  affine.for %arg1 = 0 to %51 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%50]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%50]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%48]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%48]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%46]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%46]
    %126 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%125)[%44]
    %127 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%125)[%44]
    %128 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%127)[%42]
    %129 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%127)[%42]
    %130 = affine.load %alloc_3[0, %128 * 2 + %122, %126 * 2 + %120, %124] : memref<1x28x28x8xf32>
    %131 = affine.load %3[0, %124, 0, 0] : memref<1x8x1x1xf32>
    %132 = affine.load %alloc_4[%129, %128, %126, %124] : memref<1x14x14x8xf32>
    %133 = arith.addf %130, %131 : f32
    %134 = arith.maximumf %133, %cst : f32
    %135 = arith.maximumf %132, %134 : f32
    affine.store %135, %alloc_4[%129, %128, %126, %124] : memref<1x14x14x8xf32>
  }
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %52 = affine.apply affine_map<() -> (1)>()
  %53 = affine.apply affine_map<() -> (18)>()
  %54 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%52)[%53]
  %55 = affine.apply affine_map<() -> (18)>()
  %56 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%54)[%55]
  %57 = affine.apply affine_map<() -> (8)>()
  %58 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%56)[%57]
  affine.for %arg1 = 0 to %58 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%57]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%57]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%55]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%55]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%53]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%53]
    affine.store %cst, %alloc_5[%125, %124, %122, %120] : memref<1x18x18x8xf32>
  }
  %reinterpret_cast_6 = memref.reinterpret_cast %alloc_5 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_4, %reinterpret_cast_6 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  %59 = affine.apply affine_map<() -> (1)>()
  %60 = affine.apply affine_map<() -> (14)>()
  %61 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%59)[%60]
  %62 = affine.apply affine_map<() -> (14)>()
  %63 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%61)[%62]
  %64 = affine.apply affine_map<() -> (16)>()
  %65 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%63)[%64]
  affine.for %arg1 = 0 to %65 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%64]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%64]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%62]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%62]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%60]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%60]
    affine.store %cst, %alloc_7[%125, %124, %122, %120] : memref<1x14x14x16xf32>
  }
  %66 = affine.apply affine_map<() -> (1)>()
  %67 = affine.apply affine_map<() -> (14)>()
  %68 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%66)[%67]
  %69 = affine.apply affine_map<() -> (14)>()
  %70 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%68)[%69]
  %71 = affine.apply affine_map<() -> (16)>()
  %72 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%70)[%71]
  %73 = affine.apply affine_map<() -> (5)>()
  %74 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%72)[%73]
  %75 = affine.apply affine_map<() -> (5)>()
  %76 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%74)[%75]
  %77 = affine.apply affine_map<() -> (8)>()
  %78 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%76)[%77]
  affine.for %arg1 = 0 to %78 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%77]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%77]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%75]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%75]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%73]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%73]
    %126 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%125)[%71]
    %127 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%125)[%71]
    %128 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%127)[%69]
    %129 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%127)[%69]
    %130 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%129)[%67]
    %131 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%129)[%67]
    %132 = affine.load %alloc_5[%131, %130 + %124, %128 + %122, %120] : memref<1x18x18x8xf32>
    %133 = affine.load %1[%126, %124, %122, %120] : memref<16x5x5x8xf32>
    %134 = affine.load %alloc_7[%131, %130, %128, %126] : memref<1x14x14x16xf32>
    %135 = arith.mulf %132, %133 : f32
    %136 = arith.addf %134, %135 : f32
    affine.store %136, %alloc_7[%131, %130, %128, %126] : memref<1x14x14x16xf32>
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  %79 = affine.apply affine_map<() -> (1)>()
  %80 = affine.apply affine_map<() -> (4)>()
  %81 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%79)[%80]
  %82 = affine.apply affine_map<() -> (4)>()
  %83 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%81)[%82]
  %84 = affine.apply affine_map<() -> (16)>()
  %85 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%83)[%84]
  affine.for %arg1 = 0 to %85 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%84]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%84]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%82]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%82]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%80]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%80]
    affine.store %cst_0, %alloc_8[%125, %124, %122, %120] : memref<1x4x4x16xf32>
  }
  %86 = affine.apply affine_map<() -> (1)>()
  %87 = affine.apply affine_map<() -> (4)>()
  %88 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%86)[%87]
  %89 = affine.apply affine_map<() -> (4)>()
  %90 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%88)[%89]
  %91 = affine.apply affine_map<() -> (16)>()
  %92 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%90)[%91]
  %93 = affine.apply affine_map<() -> (3)>()
  %94 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%92)[%93]
  %95 = affine.apply affine_map<() -> (3)>()
  %96 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%94)[%95]
  affine.for %arg1 = 0 to %96 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%95]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%95]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%93]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%93]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%91]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%91]
    %126 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%125)[%89]
    %127 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%125)[%89]
    %128 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%127)[%87]
    %129 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%127)[%87]
    %130 = affine.load %alloc_7[0, %128 * 3 + %122, %126 * 3 + %120, %124] : memref<1x14x14x16xf32>
    %131 = affine.load %2[0, %124, 0, 0] : memref<1x16x1x1xf32>
    %132 = affine.load %alloc_8[%129, %128, %126, %124] : memref<1x4x4x16xf32>
    %133 = arith.addf %130, %131 : f32
    %134 = arith.maximumf %133, %cst : f32
    %135 = arith.maximumf %132, %134 : f32
    affine.store %135, %alloc_8[%129, %128, %126, %124] : memref<1x4x4x16xf32>
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %97 = affine.apply affine_map<() -> (1)>()
  %98 = affine.apply affine_map<() -> (16)>()
  %99 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%97)[%98]
  %100 = affine.apply affine_map<() -> (4)>()
  %101 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%99)[%100]
  %102 = affine.apply affine_map<() -> (4)>()
  %103 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%101)[%102]
  affine.for %arg1 = 0 to %103 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%102]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%102]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%100]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%100]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%98]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%98]
    %126 = affine.load %alloc_8[%125, %122, %120, %124] : memref<1x4x4x16xf32>
    affine.store %126, %alloc_9[%125, %124, %122, %120] : memref<1x16x4x4xf32>
  }
  %reinterpret_cast_10 = memref.reinterpret_cast %alloc_9 to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<1x16x4x4xf32> to memref<1x1x256xf32>
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %104 = affine.apply affine_map<() -> (1)>()
  %105 = affine.apply affine_map<() -> (1)>()
  %106 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%104)[%105]
  %107 = affine.apply affine_map<() -> (10)>()
  %108 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%106)[%107]
  affine.for %arg1 = 0 to %108 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%107]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%107]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%105]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%105]
    affine.store %cst, %alloc_11[%123, %122, %120] : memref<1x1x10xf32>
  }
  %109 = affine.apply affine_map<() -> (1)>()
  %110 = affine.apply affine_map<() -> (1)>()
  %111 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%109)[%110]
  %112 = affine.apply affine_map<() -> (10)>()
  %113 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%111)[%112]
  %114 = affine.apply affine_map<() -> (256)>()
  %115 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%113)[%114]
  affine.for %arg1 = 0 to %115 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%114]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%114]
    %122 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%121)[%112]
    %123 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%121)[%112]
    %124 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%123)[%110]
    %125 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%123)[%110]
    %126 = affine.load %reinterpret_cast_10[%125, %124, %120] : memref<1x1x256xf32>
    %127 = affine.load %5[%125, %120, %122] : memref<1x256x10xf32>
    %128 = affine.load %alloc_11[%125, %124, %122] : memref<1x1x10xf32>
    %129 = arith.mulf %126, %127 : f32
    %130 = arith.addf %128, %129 : f32
    affine.store %130, %alloc_11[%125, %124, %122] : memref<1x1x10xf32>
  }
  %reinterpret_cast_12 = memref.reinterpret_cast %alloc_11 to offset: [0], sizes: [1, 10], strides: [10, 1] : memref<1x1x10xf32> to memref<1x10xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %116 = affine.apply affine_map<() -> (1)>()
  %117 = affine.apply affine_map<() -> (10)>()
  %118 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%116)[%117]
  affine.for %arg1 = 0 to %118 {
    %120 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%117]
    %121 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%117]
    %122 = affine.load %reinterpret_cast_12[0, %120] : memref<1x10xf32>
    %123 = affine.load %6[0, %120] : memref<1x10xf32>
    %124 = arith.addf %122, %123 : f32
    affine.store %124, %alloc_13[%121, %120] : memref<1x10xf32>
  }
  %119 = bufferization.to_tensor %alloc_13 : memref<1x10xf32>
  return %119 : tensor<1x10xf32>
}

