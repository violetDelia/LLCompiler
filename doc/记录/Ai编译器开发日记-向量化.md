# 向量化与vector dialect

向量化是相当常见的优化手段，是将多个重复且连续的单指令变为一条向量指令的优化手段。

在MLIR中，vector dialect是专门做向量化的一个dialect，在MLIR中的向量化并不是将OP变为向量指令，而是将多个标量的计算合并成很长的一个向量计算表示。这样再转到LLVM IR 去生成向量指令会更容易一些。

其实可以通过内建函数的方式直接从MLIR中变为向量指令，但是这样做不是很好，因为LLVM 后端是需要去做向量化的，没有必要再MLIR上直接变为向量指令，并且变为向量指令的话，就和后端机器过于绑定了，但其实MLIR上并不需要去做这些，这应给是LLVM IR 转为机器指令时候的职能。

 以一个简单的add为例：

```
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x + 3
        return x
```

Add模型的输入shape是【200，3，224，224】

其静态的图IR如下：

```
func.func @main(%arg0: tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c256"}) -> tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c256, value = 256 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c224, value = 224 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c200, value = 200 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %5 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %6 = "llh.constant"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %7 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %8 = "llh.add"(%arg0, %arg0) : (tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>, tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>) -> tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>
    %9 = "llh.reshape"(%6, %5, %5, %5, %5) : (tensor<1xf32, #llh.encoding<shapes = @c1>>, i64, i64, i64, i64) -> tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>
    %10 = "llh.broadcast_to"(%9, %2, %7, %1, %0) <{cast_dims = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>, i64, i64, i64, i64) -> tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>
    %11 = "llh.add"(%8, %10) : (tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>, tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>) -> tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>
    return %11 : tensor<200x3x224x256xf32, #llh.encoding<shapes = @c200, @c3, @c224, @c256>>
  }
```

## 向量化MLIR 部分示例

将所示的模型下降到向量化之前如下：

```
func.func @main(%arg0: memref<200x3x224x256xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c256"}) -> memref<200x3x224x256xf32> attributes {entrance} {
    %cst = arith.constant 3.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<200x3x224x256xf32>
    affine.for %arg1 = 0 to 200 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 224 {
          affine.for %arg4 = 0 to 256 {
            %0 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<200x3x224x256xf32>
            %1 = arith.addf %0, %0 : f32
            %2 = arith.addf %1, %cst : f32
            affine.store %2, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<200x3x224x256xf32>
          }
        }
      }
    }
    return %alloc : memref<200x3x224x256xf32>
  }
```

经过Pass affine-super-vectorize 并且指定向量大小为128，IR变换为：

```

func.func @main(%arg0: memref<200x3x224x256xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c256"}) -> memref<200x3x224x256xf32> attributes {entrance} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<3.000000e+00> : vector<128xf32>
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c224 = arith.constant 224 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c200 = arith.constant 200 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<200x3x224x256xf32>
  scf.for %arg1 = %c0 to %c200 step %c1 {
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c224 step %c1 {
        scf.for %arg4 = %c0 to %c256 step %c128 {
          %0 = vector.transfer_read %arg0[%arg1, %arg2, %arg3, %arg4], %cst : memref<200x3x224x256xf32>, vector<128xf32>
          %1 = arith.addf %0, %0 : vector<128xf32>
          %2 = arith.addf %1, %cst_0 : vector<128xf32>
          vector.transfer_write %2, %alloc[%arg1, %arg2, %arg3, %arg4] : vector<128xf32>, memref<200x3x224x256xf32>
        }
      }
    }
  }
```

如下，这里推荐最小向量化的大小是128个，因为现在的向量指令基本支持512位的，128可以变为若干条向量指令了。如果太长的话后端编译器不好做指令重排，太少的话，后端循环展开就比较麻烦了。所以推荐256~1024最好。

mlir 其实有两条vector的路线，一个是再linaga->vector->scf  , 一个是affine ->vector->scf。这里展示的是从affine ->vector的路线。

另一个是mlir是支持2d的向量化，因为现代指令都会有2d指令来计算matmul、reduce或者store、load，如果是1d的vector是不能直接变为2d的指令的，llvm后端需要做指令替换的优化Pass。如果在MLIR上做2D的向量化，那么tilling就会很难做，所以我打算只在conv\matmul\部分reduce上去做2d的向量化。

## 向量化LLVM 部分

当将IR 下降到LLVM IR 上并且优化完毕后如下：

```
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none)
define void @main(ptr nocapture readonly %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds i8, ptr %0, i64 48
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds i8, ptr %0, i64 8
  %5 = load ptr, ptr %4, align 8
  br label %.preheader2

.preheader2:                                      ; preds = %1, %49
  %6 = phi i64 [ 0, %1 ], [ %50, %49 ]
  %7 = mul nuw nsw i64 %6, 172032
  br label %.preheader

.preheader:                                       ; preds = %.preheader2, %.preheader
  %8 = phi i64 [ 0, %.preheader2 ], [ %20, %.preheader ]
  %9 = shl i64 %8, 8
  %10 = add nuw nsw i64 %7, %9
  %11 = getelementptr float, ptr %5, i64 %10
  %unmaskedload = load <128 x float>, ptr %11, align 4
  %12 = fadd <128 x float> %unmaskedload, %unmaskedload
  %13 = fadd <128 x float> %12, splat (float 3.000000e+00)
  %14 = getelementptr float, ptr %3, i64 %10
  store <128 x float> %13, ptr %14, align 4
  %15 = or disjoint i64 %10, 128
  %16 = getelementptr float, ptr %5, i64 %15
  %unmaskedload5 = load <128 x float>, ptr %16, align 4
  %17 = fadd <128 x float> %unmaskedload5, %unmaskedload5
  %18 = fadd <128 x float> %17, splat (float 3.000000e+00)
  %19 = getelementptr float, ptr %3, i64 %15
  store <128 x float> %18, ptr %19, align 4
  %20 = add nuw nsw i64 %8, 1
  %exitcond.not = icmp eq i64 %20, 224
  br i1 %exitcond.not, label %.preheader1.1, label %.preheader

.preheader1.1:                                    ; preds = %.preheader
  %21 = add nuw nsw i64 %7, 57344
  br label %.preheader.1

.preheader.1:                                     ; preds = %.preheader.1, %.preheader1.1
  %22 = phi i64 [ 0, %.preheader1.1 ], [ %34, %.preheader.1 ]
  %23 = shl i64 %22, 8
  %24 = add nuw nsw i64 %21, %23
  %25 = getelementptr float, ptr %5, i64 %24
  %unmaskedload6 = load <128 x float>, ptr %25, align 4
  %26 = fadd <128 x float> %unmaskedload6, %unmaskedload6
  %27 = fadd <128 x float> %26, splat (float 3.000000e+00)
  %28 = getelementptr float, ptr %3, i64 %24
  store <128 x float> %27, ptr %28, align 4
  %29 = or disjoint i64 %24, 128
  %30 = getelementptr float, ptr %5, i64 %29
  %unmaskedload7 = load <128 x float>, ptr %30, align 4
  %31 = fadd <128 x float> %unmaskedload7, %unmaskedload7
  %32 = fadd <128 x float> %31, splat (float 3.000000e+00)
  %33 = getelementptr float, ptr %3, i64 %29
  store <128 x float> %32, ptr %33, align 4
  %34 = add nuw nsw i64 %22, 1
  %exitcond.1.not = icmp eq i64 %34, 224
  br i1 %exitcond.1.not, label %.preheader1.2, label %.preheader.1

.preheader1.2:                                    ; preds = %.preheader.1
  %35 = add nuw nsw i64 %7, 114688
  br label %.preheader.2

.preheader.2:                                     ; preds = %.preheader.2, %.preheader1.2
  %36 = phi i64 [ 0, %.preheader1.2 ], [ %48, %.preheader.2 ]
  %37 = shl i64 %36, 8
  %38 = add nuw nsw i64 %35, %37
  %39 = getelementptr float, ptr %5, i64 %38
  %unmaskedload8 = load <128 x float>, ptr %39, align 4
  %40 = fadd <128 x float> %unmaskedload8, %unmaskedload8
  %41 = fadd <128 x float> %40, splat (float 3.000000e+00)
  %42 = getelementptr float, ptr %3, i64 %38
  store <128 x float> %41, ptr %42, align 4
  %43 = or disjoint i64 %38, 128
  %44 = getelementptr float, ptr %5, i64 %43
  %unmaskedload9 = load <128 x float>, ptr %44, align 4
  %45 = fadd <128 x float> %unmaskedload9, %unmaskedload9
  %46 = fadd <128 x float> %45, splat (float 3.000000e+00)
  %47 = getelementptr float, ptr %3, i64 %43
  store <128 x float> %46, ptr %47, align 4
  %48 = add nuw nsw i64 %36, 1
  %exitcond.2.not = icmp eq i64 %48, 224
  br i1 %exitcond.2.not, label %49, label %.preheader.2

49:                                               ; preds = %.preheader.2
  %50 = add nuw nsw i64 %6, 1
  %exitcond4.not = icmp eq i64 %50, 200
  br i1 %exitcond4.not, label %51, label %.preheader2

51:                                               ; preds = %49
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

可以看到其中主要的计算和load部分类型都变成了<128xfloat>，如果不做向量化，在LLVM IR 上是这样的：

```
 %43 = add nuw nsw i64 %40, %42
  %44 = getelementptr float, ptr %5, i64 %43
  %45 = load float, ptr %44, align 4
  %46 = fadd float %45, %45
  %47 = fadd float %46, 3.000000e+00
  %48 = getelementptr float, ptr %3, i64 %43
```

## 向量化优化提升

笔者做了一个简单的统计，如果未做向量化，element wise的平均时间是0.20s，做了向量化的平均时间是0.16s，性能提高了20%。说明在MLIR上做向量化之后确实可以让LLVM后端做更进一步的优化。

推断是因为如果LLVM IR 是为向量化的fadd float，LLVM 也会去做向量化，但是在循环体内部就是1条指令，之后在做循环展开不容易将流水充分利用起来。拿到fadd <128xfloat>，会在循环体内部变为多条向量指令，之后做循环展开就更容易利用流水了，所以性能会提升。   ps（目前还不打算做llvm 后端的部分，所以就没看转到汇编变成了啥，emm不过推测应该是这样）
