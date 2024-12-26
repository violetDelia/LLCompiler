# 前言

一般框架定义的Tensor数据结构实际上是一段连续的内存空间，根据它的size 和 stride 来决定索引，NCHW和NHWC是最常见，也是支持最广泛的结构（写算子一般也就是这两种排布了），从torch拿过来的排布默认是NCHW，但是实际计算的时候NHWC这种排布方式更适合并行计算【在当前硬件主流的三级缓存的结构下】。所以需要将前端拿到的输入转换为适合后端计算的排布方式。

## 布局表示

首先定义一个属性表示数据的排布方式：

```
def LLH_LAYOUT_NCHW         : I32EnumAttrCase<"NCHW", 0>; 
def LLH_LAYOUT_NHWC         : I32EnumAttrCase<"NHWC", 1>;//C last
def LLH_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [LLH_LAYOUT_NCHW, LLH_LAYOUT_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::llh";
}
```

然后在图中显示的表示：

```
#map = affine_map<() -> (1, 3, 7, 7)>
#map1 = affine_map<() -> (200, 3, 100, 100)>
#map2 = affine_map<() -> (200, 1, 96, 96)>
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  func.func @main(%arg0: tensor<1x3x7x7xf32> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "c7", func.input_symbol_3 = "c7"}, %arg1: tensor<200x3x100x100xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c100", func.input_symbol_3 = "c100"}) -> (tensor<200x1x96x96xf32>, tensor<1x3x7x7xf32>, tensor<200x3x100x100xf32>) attributes {entrance} {
    %0 = "llh.constant"() <{value = 96 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = 200 : i64}> : () -> i64
    "llh.symbolic_bind"(%arg0) <{expressions = #map}> : (tensor<1x3x7x7xf32>) -> ()
    "llh.symbolic_bind"(%arg1) <{expressions = #map1}> : (tensor<200x3x100x100xf32>) -> ()
    %3 = "llh.conv"(%arg1, %arg0) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<200x3x100x100xf32>, tensor<1x3x7x7xf32>) -> tensor<200x1x96x96xf32>
    "llh.symbolic_bind"(%3) <{expressions = #map2}> : (tensor<200x1x96x96xf32>) -> ()
    %4 = "llh.reshape"(%3, %2, %1, %0, %0) : (tensor<200x1x96x96xf32>, i64, i64, i64, i64) -> tensor<200x1x96x96xf32>
    "llh.symbolic_bind"(%4) <{expressions = #map2}> : (tensor<200x1x96x96xf32>) -> ()
    return %4, %arg0, %arg1 : tensor<200x1x96x96xf32>, tensor<1x3x7x7xf32>, tensor<200x3x100x100xf32>
  }
}
```
上图是一个训练的conv算子生成的计算图，上面layout表示它的布局是NCHW的，需要将其变换为NHWC的格式。以适配后端的计算设备

## 转换布局

