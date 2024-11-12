# 规范化

除了MLIR 的 Fold接口外，还有一种更为一般化的优化手段，就是“规范化”。

一般来讲规范化的用在以下两个方面：

1. 在不修改语义的情况下调整IR，使之后的分析和优化更容易进行。
2. 进行通用化的优化手段。

但是需要注意的是：
定义的规范化方法不能无限重复的匹配，连续匹配若干次（默认是10次，可自行设置）后会导致优化停止。

## 定义方法

在MLIR中，每一个Op都有属于自己的规范化方法，如果想要给某个Op定义其规范化方法，可在定义的Op上添加hasCanonicalizeMethod或者hasCanonicalizer。
推荐使用hasCanonicalizeMethod。

```
def LLH_DimOp : LLH_Op<"dim">{
    let arguments = (ins 
        LLH_Tensor:$input,
        LLH_Int64:$dim,
        OptionalAttr<FlatSymbolRefAttr>:$symbol
        );
    let results = (outs 
        LLH_Int64);
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}
```

然后在实现它的getCanonicalizationPatterns方法就可以了。

```
namespace {
struct DimOpToConst : public LLHOpRewritePattern<DimOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(DimOp op) const final {
    auto input = op.getInput(); //获取DimOp的输入
    if (!isa<RankedTensorType>(input.getType())) return llvm::failure();
    auto maybe_const_dim = op.getDim();
    if (!llh::isConstIntegerValue(maybe_const_dim)) return llvm::failure();
    auto type = llc::getRankTensorFrom(input);
    auto dim = llh::getConstIntegerValue(maybe_const_dim); 
    if (type.isDynamicDim(dim)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(DimOp op, LLHPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto value = llh::getConstIntegerValue(op);
    auto new_op = rewriter.replaceOpWithNewOp<ConstantOp>(
        op, rewriter.getI64IntegerAttr(value));
  }
};
}  // namespace
void DimOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DimOpToConst>(context);
}
```

## 示例
上文给出的示例是一个DimOp的规范化方法，如果DimOp的值是一个常数的化，会将DimOp改写为ConstOp。

规范化之前：
```
func.func @dim_to_const(%101: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) ->(i64, i64, i64, i64) attributes {entrance} {
  %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
  %102 = "llh.dim"(%101, %2) <{symbol = @s0}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %103 = "llh.dim"(%101, %3) <{symbol = @c512}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %104 = "llh.dim"(%101, %1) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %105 = "llh.dim"(%101, %0) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  return %102,%103,%104,%105: i64, i64, i64, i64
}
```
规范化之后：
```
module {
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @dim_to_const(%arg0: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) -> (i64, i64, i64, i64) attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c512, value = 512 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %5 = "llh.dim"(%arg0, %3) <{symbol = @s0}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    return %5, %0, %4, %4 : i64, i64, i64, i64
  }
}
```
可以看到除了%5的tensor第一个维度是动态没有被转换以外，其他的llh.dim都转变成看常量llh.constant.
