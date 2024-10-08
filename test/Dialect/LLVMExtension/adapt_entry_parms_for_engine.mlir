// RUN: llc-opt --split-input-file --adapt-entry-parms-for-engine %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --adapt-entry-parms-for-engine /home/lfr/LLCompiler/test/Dialect/LLVMExtension/adapt_entry_parms_for_engine.mlir
module attributes {builtin.gloabal_layout = "NCHW"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  // CHECK-LABEL: main
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {entrance} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.mlir.constant(4 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %8 = llvm.getelementptr %1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.add %9, %0 : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.sub %0, %5 : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.urem %14, %0  : i64
    %16 = llvm.sub %14, %15 : i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %18 = llvm.insertvalue %11, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %17, %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %6, %19[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %2, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %5, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %4, %22[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %4, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %4, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %5, %25[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%27: i64):  // 2 preds: ^bb0, ^bb2
    %28 = llvm.icmp "slt" %27, %3 : i64
    llvm.cond_br %28, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %29 = llvm.urem %27, %4  : i64
    %30 = llvm.udiv %27, %4  : i64
    %31 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.mul %30, %arg6 : i64
    %33 = llvm.mul %arg7, %6 : i64
    %34 = llvm.add %32, %33 : i64
    %35 = llvm.mul %29, %arg8 : i64
    %36 = llvm.add %34, %35 : i64
    %37 = llvm.getelementptr %31[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.fadd %38, %38  : f32
    %40 = llvm.mul %30, %4 : i64
    %41 = llvm.mul %6, %4 : i64
    %42 = llvm.add %40, %41 : i64
    %43 = llvm.add %42, %29 : i64
    %44 = llvm.getelementptr %17[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %39, %44 : f32, !llvm.ptr
    %45 = llvm.add %27, %5 : i64
    llvm.br ^bb1(%45 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return %26 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}