// -----// IR Dump After BufferLoopHoisting (buffer-loop-hoisting) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x10xf32> {
  %c16 = arith.constant 16 : index
  %c10 = arith.constant 10 : index
  %c-3 = arith.constant -3 : index
  %c-12 = arith.constant -12 : index
  %c3 = arith.constant 3 : index
  %c200 = arith.constant 200 : index
  %c9 = arith.constant 9 : index
  %c256 = arith.constant 256 : index
  %c18 = arith.constant 18 : index
  %c2592 = arith.constant 2592 : index
  %c-5 = arith.constant -5 : index
  %c-2 = arith.constant -2 : index
  %c-28 = arith.constant -28 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c25 = arith.constant 25 : index
  %c4 = arith.constant 4 : index
  %c14 = arith.constant 14 : index
  %c8 = arith.constant 8 : index
  %c1568 = arith.constant 1568 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
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
  scf.parallel (%arg1) = (%c0) to (%c1024) step (%c1) {
    %6 = arith.remui %arg1, %c32 : index
    %7 = arith.divui %arg1, %c32 : index
    memref.store %cst_0, %alloc_4[%c0, %7, %6, %c0] : memref<1x32x32x1xf32>
    scf.reduce 
  }
  %subview = memref.subview %alloc_4[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  scf.for %arg1 = %c0 to %c1568 step %c1 {
    %6 = arith.remui %arg1, %c8 : index
    %7 = arith.divui %arg1, %c8 : index
    %8 = arith.remui %7, %c14 : index
    %9 = arith.divui %arg1, %c8 : index
    %10 = arith.divui %9, %c14 : index
    memref.store %cst, %alloc_5[%c0, %10, %8, %6] : memref<1x14x14x8xf32>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      memref.store %cst_0, %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      scf.for %arg3 = %c0 to %c25 step %c1 {
        %23 = arith.remui %arg3, %c5 : index
        %24 = arith.divui %arg3, %c5 : index
        %25 = arith.divui %arg1, %c8 : index
        %26 = arith.divui %25, %c14 : index
        %27 = arith.muli %26, %c2 : index
        %28 = arith.divui %arg2, %c2 : index
        %29 = arith.addi %27, %28 : index
        %30 = arith.divui %arg3, %c5 : index
        %31 = arith.addi %29, %30 : index
        %32 = arith.addi %arg3, %arg2 : index
        %33 = arith.divui %arg1, %c8 : index
        %34 = arith.muli %33, %c2 : index
        %35 = arith.addi %32, %34 : index
        %36 = arith.divui %arg1, %c8 : index
        %37 = arith.divui %36, %c14 : index
        %38 = arith.muli %37, %c-28 : index
        %39 = arith.addi %35, %38 : index
        %40 = arith.divui %arg2, %c2 : index
        %41 = arith.muli %40, %c-2 : index
        %42 = arith.addi %39, %41 : index
        %43 = arith.divui %arg3, %c5 : index
        %44 = arith.muli %43, %c-5 : index
        %45 = arith.addi %42, %44 : index
        %46 = memref.load %alloc_4[%c0, %31, %45, %c0] : memref<1x32x32x1xf32>
        %47 = memref.load %3[%6, %24, %23, %c0] : memref<8x5x5x1xf32>
        %48 = memref.load %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
        %49 = arith.mulf %46, %47 : f32
        %50 = arith.addf %48, %49 : f32
        memref.store %50, %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      }
      %11 = memref.load %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %12 = memref.load %2[%c0, %6, %c0, %c0] : memref<1x8x1x1xf32>
      %13 = memref.load %alloc_5[%c0, %10, %8, %6] : memref<1x14x14x8xf32>
      %14 = arith.addf %11, %12 : f32
      %15 = arith.cmpf ugt, %14, %cst_0 : f32
      %16 = arith.select %15, %14, %cst_0 : f32
      %17 = arith.cmpf uno, %cst_0, %cst_0 : f32
      %18 = arith.select %17, %cst_0, %16 : f32
      %19 = arith.cmpf ugt, %13, %18 : f32
      %20 = arith.select %19, %13, %18 : f32
      %21 = arith.cmpf uno, %18, %18 : f32
      %22 = arith.select %21, %18, %20 : f32
      memref.store %22, %alloc_5[%c0, %10, %8, %6] : memref<1x14x14x8xf32>
    }
  }
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  scf.parallel (%arg1) = (%c0) to (%c2592) step (%c1) {
    %6 = arith.remui %arg1, %c8 : index
    %7 = arith.divui %arg1, %c8 : index
    %8 = arith.remui %7, %c18 : index
    %9 = arith.divui %arg1, %c8 : index
    %10 = arith.divui %9, %c18 : index
    memref.store %cst_0, %alloc_6[%c0, %10, %8, %6] : memref<1x18x18x8xf32>
    scf.reduce 
  }
  %subview_7 = memref.subview %alloc_6[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_5, %subview_7 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  scf.for %arg1 = %c0 to %c256 step %c1 {
    %6 = arith.remui %arg1, %c4 : index
    %7 = arith.divui %arg1, %c4 : index
    %8 = arith.remui %7, %c4 : index
    %9 = arith.divui %arg1, %c4 : index
    %10 = arith.divui %9, %c4 : index
    memref.store %cst, %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    scf.for %arg2 = %c0 to %c9 step %c1 {
      memref.store %cst_0, %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      scf.for %arg3 = %c0 to %c200 step %c1 {
        %24 = arith.remui %arg3, %c8 : index
        %25 = arith.divui %arg3, %c8 : index
        %26 = arith.remui %25, %c5 : index
        %27 = arith.divui %arg3, %c8 : index
        %28 = arith.divui %27, %c5 : index
        %29 = arith.divui %arg1, %c4 : index
        %30 = arith.muli %29, %c3 : index
        %31 = arith.divui %arg1, %c4 : index
        %32 = arith.divui %31, %c4 : index
        %33 = arith.muli %32, %c-12 : index
        %34 = arith.addi %30, %33 : index
        %35 = arith.divui %arg2, %c3 : index
        %36 = arith.addi %34, %35 : index
        %37 = arith.divui %arg3, %c8 : index
        %38 = arith.divui %37, %c5 : index
        %39 = arith.addi %36, %38 : index
        %40 = arith.muli %arg1, %c3 : index
        %41 = arith.addi %40, %arg2 : index
        %42 = arith.divui %arg1, %c4 : index
        %43 = arith.muli %42, %c-12 : index
        %44 = arith.addi %41, %43 : index
        %45 = arith.divui %arg2, %c3 : index
        %46 = arith.muli %45, %c-3 : index
        %47 = arith.addi %44, %46 : index
        %48 = arith.divui %arg3, %c8 : index
        %49 = arith.addi %47, %48 : index
        %50 = arith.divui %arg3, %c8 : index
        %51 = arith.divui %50, %c5 : index
        %52 = arith.muli %51, %c-5 : index
        %53 = arith.addi %49, %52 : index
        %54 = memref.load %alloc_6[%c0, %39, %53, %24] : memref<1x18x18x8xf32>
        %55 = memref.load %0[%10, %28, %26, %24] : memref<16x5x5x8xf32>
        %56 = memref.load %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
        %57 = arith.mulf %54, %55 : f32
        %58 = arith.addf %56, %57 : f32
        memref.store %58, %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      }
      %12 = memref.load %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %13 = memref.load %1[%c0, %10, %c0, %c0] : memref<1x16x1x1xf32>
      %14 = memref.load %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %15 = arith.addf %12, %13 : f32
      %16 = arith.cmpf ugt, %15, %cst_0 : f32
      %17 = arith.select %16, %15, %cst_0 : f32
      %18 = arith.cmpf uno, %cst_0, %cst_0 : f32
      %19 = arith.select %18, %cst_0, %17 : f32
      %20 = arith.cmpf ugt, %14, %19 : f32
      %21 = arith.select %20, %14, %19 : f32
      %22 = arith.cmpf uno, %19, %19 : f32
      %23 = arith.select %22, %19, %21 : f32
      memref.store %23, %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    }
    %11 = memref.load %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    memref.store %11, %alloc_8[%c0, %10, %8, %6] : memref<1x16x4x4xf32>
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  scf.parallel (%arg1) = (%c0) to (%c10) step (%c1) {
    memref.store %cst_0, %alloc_9[%c0, %c0, %arg1] : memref<1x1x10xf32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %6 = arith.divui %arg2, %c16 : index
      %7 = arith.remui %arg2, %c16 : index
      %8 = arith.divui %7, %c4 : index
      %9 = arith.remui %arg2, %c4 : index
      %10 = memref.load %alloc_8[%c0, %6, %8, %9] : memref<1x16x4x4xf32>
      %11 = memref.load %4[%c0, %arg2, %arg1] : memref<1x256x10xf32>
      %12 = memref.load %alloc_9[%c0, %c0, %arg1] : memref<1x1x10xf32>
      %13 = arith.mulf %10, %11 : f32
      %14 = arith.addf %12, %13 : f32
      memref.store %14, %alloc_9[%c0, %c0, %arg1] : memref<1x1x10xf32>
    }
    scf.reduce 
  }
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  scf.parallel (%arg1) = (%c0) to (%c10) step (%c1) {
    %6 = memref.load %alloc_9[%c0, %c0, %arg1] : memref<1x1x10xf32>
    %7 = memref.load %5[%c0, %arg1] : memref<1x10xf32>
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %alloc_10[%c0, %arg1] : memref<1x10xf32>
    scf.reduce 
  }
  return %alloc_10 : memref<1x10xf32>
}

