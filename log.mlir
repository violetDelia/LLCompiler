module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: memref<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}, %arg1: memref<?x?x?x?xf32>) attributes {entrance} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<?x?x?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<?x?x?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %cst = arith.constant dense<3.000000e+00> : vector<128xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %dim_3 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
    %alloc = memref.alloc(%dim, %dim_1, %dim_2, %dim_3) {alignment = 64 : i64} : memref<?x?x?x?xf32>
    %2 = builtin.unrealized_conversion_cast %alloc : memref<?x?x?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb11
    %4 = builtin.unrealized_conversion_cast %3 : index to i64
    %5 = arith.cmpi slt, %3, %dim : index
    cf.cond_br %5, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb10
    %7 = builtin.unrealized_conversion_cast %6 : index to i64
    %8 = arith.cmpi slt, %6, %dim_1 : index
    cf.cond_br %8, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c0 : index)
  ^bb5(%9: index):  // 2 preds: ^bb4, ^bb9
    %10 = builtin.unrealized_conversion_cast %9 : index to i64
    %11 = arith.cmpi slt, %9, %dim_2 : index
    cf.cond_br %11, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%c0 : index)
  ^bb7(%12: index):  // 2 preds: ^bb6, ^bb8
    %13 = builtin.unrealized_conversion_cast %12 : index to i64
    %14 = arith.cmpi slt, %12, %dim_3 : index
    cf.cond_br %14, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %c3_4 = arith.constant 3 : index
    %dim_5 = memref.dim %arg0, %c3_4 : memref<?x?x?x?xf32>
    %15 = arith.subi %dim_5, %12 : index
    %cst_6 = arith.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : vector<128xi32>
    %16 = arith.index_cast %15 : index to i32
    %17 = llvm.mlir.undef : vector<128xi32>
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.insertelement %16, %17[%18 : i32] : vector<128xi32>
    %20 = llvm.shufflevector %19, %17 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<128xi32> 
    %21 = arith.cmpi slt, %cst_6, %20 : vector<128xi32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %22 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.extractvalue %1[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mul %4, %23 : i64
    %25 = llvm.extractvalue %1[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.mul %7, %25 : i64
    %27 = llvm.add %24, %26 : i64
    %28 = llvm.extractvalue %1[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.mul %10, %28 : i64
    %30 = llvm.add %27, %29 : i64
    %31 = llvm.add %30, %13 : i64
    %32 = llvm.getelementptr %22[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %33 = llvm.intr.masked.load %32, %21, %cst_7 {alignment = 4 : i32} : (!llvm.ptr, vector<128xi1>, vector<128xf32>) -> vector<128xf32>
    %34 = arith.addf %33, %33 : vector<128xf32>
    %c3_8 = arith.constant 3 : index
    %35 = arith.subi %dim_3, %12 : index
    %cst_9 = arith.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : vector<128xi32>
    %36 = arith.index_cast %35 : index to i32
    %37 = llvm.mlir.undef : vector<128xi32>
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.insertelement %36, %37[%38 : i32] : vector<128xi32>
    %40 = llvm.shufflevector %39, %37 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<128xi32> 
    %41 = arith.cmpi slt, %cst_9, %40 : vector<128xi32>
    %42 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.extractvalue %2[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.mul %4, %43 : i64
    %45 = llvm.extractvalue %2[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.mul %7, %45 : i64
    %47 = llvm.add %44, %46 : i64
    %48 = llvm.extractvalue %2[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.mul %10, %48 : i64
    %50 = llvm.add %47, %49 : i64
    %51 = llvm.add %50, %13 : i64
    %52 = llvm.getelementptr %42[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.intr.masked.store %34, %52, %41 {alignment = 4 : i32} : vector<128xf32>, vector<128xi1> into !llvm.ptr
    %53 = arith.addi %12, %c128 : index
    cf.br ^bb7(%53 : index)
  ^bb9:  // pred: ^bb7
    %54 = arith.addi %9, %c1 : index
    cf.br ^bb5(%54 : index)
  ^bb10:  // pred: ^bb5
    %55 = arith.addi %6, %c1 : index
    cf.br ^bb3(%55 : index)
  ^bb11:  // pred: ^bb3
    %56 = arith.addi %3, %c1 : index
    cf.br ^bb1(%56 : index)
  ^bb12:  // pred: ^bb1
    cf.br ^bb13(%c0 : index)
  ^bb13(%57: index):  // 2 preds: ^bb12, ^bb23
    %58 = builtin.unrealized_conversion_cast %57 : index to i64
    %59 = arith.cmpi slt, %57, %dim : index
    cf.cond_br %59, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    cf.br ^bb15(%c0 : index)
  ^bb15(%60: index):  // 2 preds: ^bb14, ^bb22
    %61 = builtin.unrealized_conversion_cast %60 : index to i64
    %62 = arith.cmpi slt, %60, %dim_1 : index
    cf.cond_br %62, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    cf.br ^bb17(%c0 : index)
  ^bb17(%63: index):  // 2 preds: ^bb16, ^bb21
    %64 = builtin.unrealized_conversion_cast %63 : index to i64
    %65 = arith.cmpi slt, %63, %dim_2 : index
    cf.cond_br %65, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    cf.br ^bb19(%c0 : index)
  ^bb19(%66: index):  // 2 preds: ^bb18, ^bb20
    %67 = builtin.unrealized_conversion_cast %66 : index to i64
    %68 = arith.cmpi slt, %66, %dim_3 : index
    cf.cond_br %68, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %c3_10 = arith.constant 3 : index
    %69 = arith.subi %dim_3, %66 : index
    %cst_11 = arith.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : vector<128xi32>
    %70 = arith.index_cast %69 : index to i32
    %71 = llvm.mlir.undef : vector<128xi32>
    %72 = llvm.mlir.constant(0 : i32) : i32
    %73 = llvm.insertelement %70, %71[%72 : i32] : vector<128xi32>
    %74 = llvm.shufflevector %73, %71 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<128xi32> 
    %75 = arith.cmpi slt, %cst_11, %74 : vector<128xi32>
    %cst_12 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %76 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %77 = llvm.extractvalue %2[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %78 = llvm.mul %58, %77 : i64
    %79 = llvm.extractvalue %2[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %80 = llvm.mul %61, %79 : i64
    %81 = llvm.add %78, %80 : i64
    %82 = llvm.extractvalue %2[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %83 = llvm.mul %64, %82 : i64
    %84 = llvm.add %81, %83 : i64
    %85 = llvm.add %84, %67 : i64
    %86 = llvm.getelementptr %76[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %87 = llvm.intr.masked.load %86, %75, %cst_12 {alignment = 4 : i32} : (!llvm.ptr, vector<128xi1>, vector<128xf32>) -> vector<128xf32>
    %88 = arith.addf %87, %cst : vector<128xf32>
    %c3_13 = arith.constant 3 : index
    %dim_14 = memref.dim %arg1, %c3_13 : memref<?x?x?x?xf32>
    %89 = arith.subi %dim_14, %66 : index
    %cst_15 = arith.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : vector<128xi32>
    %90 = arith.index_cast %89 : index to i32
    %91 = llvm.mlir.undef : vector<128xi32>
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = llvm.insertelement %90, %91[%92 : i32] : vector<128xi32>
    %94 = llvm.shufflevector %93, %91 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<128xi32> 
    %95 = arith.cmpi slt, %cst_15, %94 : vector<128xi32>
    %96 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %98 = llvm.mul %58, %97 : i64
    %99 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %100 = llvm.mul %61, %99 : i64
    %101 = llvm.add %98, %100 : i64
    %102 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %103 = llvm.mul %64, %102 : i64
    %104 = llvm.add %101, %103 : i64
    %105 = llvm.add %104, %67 : i64
    %106 = llvm.getelementptr %96[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.intr.masked.store %88, %106, %95 {alignment = 4 : i32} : vector<128xf32>, vector<128xi1> into !llvm.ptr
    %107 = arith.addi %66, %c128 : index
    cf.br ^bb19(%107 : index)
  ^bb21:  // pred: ^bb19
    %108 = arith.addi %63, %c1 : index
    cf.br ^bb17(%108 : index)
  ^bb22:  // pred: ^bb17
    %109 = arith.addi %60, %c1 : index
    cf.br ^bb15(%109 : index)
  ^bb23:  // pred: ^bb15
    %110 = arith.addi %57, %c1 : index
    cf.br ^bb13(%110 : index)
  ^bb24:  // pred: ^bb13
    memref.dealloc %alloc : memref<?x?x?x?xf32>
    return
  }
  module @__symbol__ {
  }
}

