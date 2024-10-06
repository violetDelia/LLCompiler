; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i32) local_unnamed_addr

define { ptr, ptr, i32, [4 x i32], [4 x i32] } @main(ptr nocapture readnone %0, ptr nocapture readonly %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, ptr nocapture readnone %11, ptr nocapture readonly %12, i32 %13, i32 %14, i32 %15, i32 %16, i32 %17, i32 %18, i32 %19, i32 %20, i32 %21) local_unnamed_addr {
  %23 = tail call ptr @malloc(i32 602176)
  %24 = ptrtoint ptr %23 to i64
  %25 = add i64 %24, 63
  %26 = and i64 %25, 4294967232
  %27 = inttoptr i64 %26 to ptr
  %28 = sext i32 %2 to i64
  %29 = getelementptr float, ptr %1, i64 %28
  %30 = sext i32 %13 to i64
  %31 = getelementptr float, ptr %12, i64 %30
  br label %32

32:                                               ; preds = %22, %32
  %33 = phi i32 [ 0, %22 ], [ %61, %32 ]
  %.frozen = freeze i32 %33
  %34 = udiv i32 %.frozen, 224
  %35 = mul i32 %34, 224
  %.decomposed = sub i32 %.frozen, %35
  %.lhs.trunc = trunc nuw i32 %34 to i16
  %36 = urem i16 %.lhs.trunc, 224
  %.zext = zext nneg i16 %36 to i32
  %37 = udiv i32 %33, 50176
  %38 = mul i32 %37, %8
  %39 = mul i32 %9, %.zext
  %40 = mul i32 %.decomposed, %10
  %41 = add i32 %40, %38
  %42 = add i32 %41, %39
  %43 = sext i32 %42 to i64
  %44 = getelementptr float, ptr %29, i64 %43
  %45 = load float, ptr %44, align 4
  %46 = mul i32 %37, %19
  %47 = mul i32 %20, %.zext
  %48 = mul i32 %.decomposed, %21
  %49 = add i32 %48, %46
  %50 = add i32 %49, %47
  %51 = sext i32 %50 to i64
  %52 = getelementptr float, ptr %31, i64 %51
  %53 = load float, ptr %52, align 4
  %54 = fadd float %45, %53
  %55 = mul nuw nsw i32 %37, 50176
  %56 = mul nuw nsw i32 %.zext, 224
  %57 = or disjoint i32 %55, %.decomposed
  %58 = add nuw nsw i32 %57, %56
  %59 = zext nneg i32 %58 to i64
  %60 = getelementptr float, ptr %27, i64 %59
  store float %54, ptr %60, align 4
  %61 = add nuw nsw i32 %33, 1
  %exitcond.not = icmp eq i32 %61, 150528
  br i1 %exitcond.not, label %62, label %32

62:                                               ; preds = %32
  %63 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } undef, ptr %23, 0
  %64 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %63, ptr %27, 1
  %65 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %64, i32 0, 2
  %66 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %65, i32 1, 3, 0
  %67 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %66, i32 3, 3, 1
  %68 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %67, i32 224, 3, 2
  %69 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %68, i32 224, 3, 3
  %70 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %69, i32 150528, 4, 0
  %71 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %70, i32 50176, 4, 1
  %72 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %71, i32 224, 4, 2
  %73 = insertvalue { ptr, ptr, i32, [4 x i32], [4 x i32] } %72, i32 1, 4, 3
  ret { ptr, ptr, i32, [4 x i32], [4 x i32] } %73
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
