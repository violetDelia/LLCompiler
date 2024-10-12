// RUN: llc-opt --split-input-file --adapt-entry-parms-for-engine %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --adapt-entry-parms-for-engine /home/lfr/LLCompiler/test/Dialect/LLVMExtension/adapt_entry_parms_for_engine.mlir
module attributes {builtin.gloabal_layout = "NCHW"} {
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) attributes {entrance} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg11, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg12, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg13, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg14, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg18, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg15, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg19, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg16, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg20, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg17, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg21, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.insertvalue %arg0, %12[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.insertvalue %arg1, %13[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg2, %14[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg3, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg7, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg4, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg8, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg5, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg9, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg6, %21[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg10, %22[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.constant(2 : index) : i64
    %25 = llvm.mlir.constant(0 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(4 : index) : i64
    %28 = llvm.mlir.constant(16 : index) : i64
    llvm.br ^bb1(%25 : i64)
  ^bb1(%29: i64):  // 2 preds: ^bb0, ^bb2
    %30 = llvm.icmp "slt" %29, %28 : i64
    llvm.cond_br %30, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %31 = llvm.urem %29, %27  : i64
    %32 = llvm.udiv %29, %27  : i64
    %33 = llvm.urem %32, %24  : i64
    %34 = llvm.udiv %32, %24  : i64
    %35 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.mlir.constant(8 : index) : i64
    %37 = llvm.mul %34, %36 : i64
    %38 = llvm.mlir.constant(4 : index) : i64
    %39 = llvm.mul %33, %38 : i64
    %40 = llvm.add %37, %39 : i64
    %41 = llvm.mlir.constant(4 : index) : i64
    %42 = llvm.mul %25, %41 : i64
    %43 = llvm.add %40, %42 : i64
    %44 = llvm.add %43, %31 : i64
    %45 = llvm.getelementptr %35[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.load %45 : !llvm.ptr -> f32
    %47 = llvm.fadd %46, %46  : f32
    %48 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.mlir.constant(8 : index) : i64
    %50 = llvm.mul %34, %49 : i64
    %51 = llvm.mlir.constant(4 : index) : i64
    %52 = llvm.mul %33, %51 : i64
    %53 = llvm.add %50, %52 : i64
    %54 = llvm.mlir.constant(4 : index) : i64
    %55 = llvm.mul %25, %54 : i64
    %56 = llvm.add %53, %55 : i64
    %57 = llvm.add %56, %31 : i64
    %58 = llvm.getelementptr %48[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %47, %58 : f32, !llvm.ptr
    %59 = llvm.add %29, %26 : i64
    llvm.br ^bb1(%59 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}