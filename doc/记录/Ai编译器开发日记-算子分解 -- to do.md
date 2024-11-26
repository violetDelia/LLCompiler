
## 引言

常量折叠是编译器基本的一种优化手段，它可以简化计算的表达式，将能够在编译器计算出结果的表达式替换为常量，从而减少计算时的开销，同时也节省内存资源。

在计算图变换的过程中会不断的出现可以进行常量折叠的表达式，如果编译器不能时刻保持进行常量折叠的状态，就会影响后期的优化。这其实是很”头疼“的事情，经常出现这样的情况：计算子图上，将一个复杂的Op变换为几个简单的Op组合，比如将softmax(x) 分解为 exp（x）/reduce_sum（exp（x）），在分解之前其实已经运行过了常量折叠的Pass，如果每次图变换都要进行一次常量折叠的Pass，会使Pass冗长且不易维护，但是不进行常量折叠，又会影响之后的优化。在MLIR框架中，提供了Fold的接口，会在每一次创建Op时检测是否可以进行折叠，提供了很大的便利性。

## MLIR框架常量折叠接口

以一个MulOp为例，介绍如何在MLIR中实现常量折叠的功能：

Mul常见的折叠方案有：

mul（x，1） =  x；

mul（x，0）= 0；

在定义Op时定义 hasFolder 指定它是否能够常量折叠：

`def LLH_MulOp           : LLH_BinaryElementwiseOp<"mul", [ResultsBroadcastableShape], LLH_Computable_Type, LLH_Computable_Type, (ins OptionalAttr<FlatSymbolRefAttr>:$symbol)>{     `

`	let hasVerifier = 1;     `

`	let hasFolder = 1; `

`}`

上述时MulOp的定义，指定 `let hasFolder = 1; `表示其能够常量折叠。

之后实现fold方法：

```

namespace {
//检测tensor张量是否为0
static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  }
  if (llvm::isa<IntegerType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  }
  return false;
}
//检测tensor张量是否为1
static bool isSplatOne(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType)) {
    return val && val.isSplat() &&
           (val.getSplatValue<APFloat>().convertToDouble() == 1);
  }
  if (llvm::isa<IntegerType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APInt>().isAllOnes();
  }
  return false;
}
//tensor数据折叠函数
DenseElementsAttr splatDenseBinaryFolder(
    DenseElementsAttr lhs, DenseElementsAttr rhs, RankedTensorType returnTy,
    function_ref<APInt(llvm::APInt, llvm::APInt)> int_calculate,
    function_ref<APFloat(llvm::APFloat, llvm::APFloat)> float_calculate) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    auto lhs_ele_type = llvm::cast<ShapedType>(lhs.getType()).getElementType();
    auto rhs_ele_type = llvm::cast<ShapedType>(rhs.getType()).getElementType();
    if (lhs_ele_type != rhs_ele_type) return {};
    if (llvm::isa<IntegerType>(lhs_ele_type)) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();
      auto result = int_calculate(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
    if (llvm::isa<FloatType>(lhs_ele_type)) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      auto result = float_calculate(l, r);
      auto c = DenseElementsAttr::get(returnTy, result);
      return DenseElementsAttr::get(returnTy, result);
    }
  }
  return {};
}
}  // namespace

//mul op 折叠实现
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto res_type = getType();
  if (!isa<IntegerType, FloatType, RankedTensorType>(res_type)) return {};
  if (isa<IntegerType>(res_type)) {
    // mul(x, 0) -> 0
    if (matchPattern(adaptor.getRhs(), m_Zero())) return getRhs();
    // mul(x, 1) -> x
    if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();
    return constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APInt &a, const APInt &b) { return a * b; });
  }
  if (isa<FloatType>(res_type)) {
    // mul(x, 0) -> 0
    if (matchPattern(adaptor.getRhs(), m_AnyZeroFloat())) return getRhs();
    // mul(x, 1) -> x
    if (matchPattern(adaptor.getRhs(), m_OneFloat())) return getLhs();
    return constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APFloat &a, const APFloat &b) { return a * b; });
  }
  if (isa<RankedTensorType>(res_type)) {
    auto lhs_type = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhs_type = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
    auto result_type = llvm::dyn_cast<RankedTensorType>(getType());
    if (!lhs_type.getElementType().isIntOrIndexOrFloat() ||
        !rhs_type.getElementType().isIntOrIndexOrFloat())
      return {};
    auto lhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getLhs());
    auto rhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getRhs());
    // mul(x, 0) -> 0
    if (lhs_type == result_type &&
        isSplatZero(result_type.getElementType(), rhs_attr))
      return getRhs();
    // mul(x, 1) -> x
    if (lhs_type == result_type &&
        isSplatOne(result_type.getElementType(), rhs_attr))
      return getLhs();
    if (!lhs_attr || !rhs_attr) return {};
    return splatDenseBinaryFolder(
        lhs_attr, rhs_attr, result_type,
        [](const APInt &a, const APInt &b) { return a * b; },
        [](const APFloat &a, const APFloat &b) { return a / b; });
  }
  return {};
};
```

仅需要实现fold方法，之后每次运行pass会对Op检测其是否实现的Fold方法，以及是否符合Fold的条件，如果符合的话会自动的帮我们进行折叠。但是这种方式只能生成从该Op获取到的属性信息进行折叠，而且不能创建新的Op。如果想要进行更复杂的变换优化手段，需要实现规范化方法或者自己单独写一个Pass。

现在测试一下吧：

```
func.func @mul_fold() ->(i64, f32, tensor<64xf32>, tensor<64xf32>) attributes {entrance} {
  %i1 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %i2 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %f2 = "llh.constant"() <{value = 2. : f32}> : () -> f32
  %f1 = "llh.constant"() <{value = 1. : f32}> : () -> f32
  %tensor_1 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %tensor = "llh.constant"() <{value = dense<2.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %tensor_0 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %mul_0 = "llh.mul"(%i2, %i1): (i64, i64) -> i64
  %mul_1 = "llh.mul"(%f2, %f1): (f32, f32) -> f32
  %mul_tensor_0 = "llh.mul"(%tensor, %tensor_0): (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %mul_tensor_1 = "llh.mul"(%tensor, %tensor_1): (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %mul_0, %mul_1,%mul_tensor_1,%mul_tensor_0 : i64, f32, tensor<64xf32>, tensor<64xf32>
}
```

折叠之后的效果：

```
module {
  func.func @mul_fold() -> (i64, f32, tensor<64xf32>, tensor<64xf32>) attributes {entrance} {
    %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
    %3 = "llh.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %4 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %6 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    return %1, %2, %5, %6 : i64, f32, tensor<64xf32>, tensor<64xf32>
  }
}
```

ok，图中所有的mul计算都被消除了，虽然现在模型大部分是以transformer为主的计算，但是图中很多关于shape的计算是可以被折叠掉的。
