// RUN: llc-opt --split-input-file --adapt-entry-parms-for-engine %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --adapt-entry-parms-for-engine /home/lfr/LLCompiler/test/Dialect/LLVMExtension/adapt_entry_parms_for_engine.mlir


// CHECK: llvm.func @main(%arg0: !llvm.ptr) attributes {entrance}
// CHECK-COUNT-23: llvm.load
llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) attributes {entrance} {
    llvm.return
}
// -----
// CHECK: llvm.func @main(%arg0: !llvm.ptr) attributes {entrance}
// CHECK-COUNT-28: llvm.getelementptr
llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) attributes {entrance} {
    llvm.return
}


    
   