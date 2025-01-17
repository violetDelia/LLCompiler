# &emsp;符号输入

    在一些场景下，需要将一个整图切分为一些子图，一方面是为了能够增加编译的时间，一方面可以降低内存的占用。在torch框架中天然支持这种功能。利用torch的OperatorSupport 和 CapabilityBasedPartitioner 功能可以将一个fx 计算图中的多个节点融合为一个计算子图。这样做的好处是编译的时候可以规避一些编译器不支持的op。

以下列模型为例：

```python

class ReduceFusion1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rule = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x - 3
        x = x / 2
        x= torch.softmax(x,1)
        x = torch.matmul(x,x.transpose(-2, -1))
        x = x + x
        x = x - 3
        x = x / 2
        return x
```

如果设置将 exp 计算设为编译器不支持的计算，原始的计算图会被融合为两个子图和一个exp op 的调用：

```
// 主计算图
opcode         name     target            args              kwargs
-------------  -------  ----------------  ----------------  --------
placeholder    arg0_1   arg0_1            ()                {}
placeholder    arg1_1   arg1_1            ()                {}
call_module    fused_1  fused_1           (arg1_1, arg0_1)  {}
call_function  exp      aten.exp.default  (fused_1,)        {}
call_module    fused_0  fused_0           (exp, arg0_1)     {}
output         output   output            ((fused_0,),)     {}

// 子图1
opcode         name                target                          args                                            kwargs
-------------  ------------------  ------------------------------  ----------------------------------------------  --------
placeholder    exp                 exp                             ()                                              {}
call_function  sum_1               aten.sum.dim_IntList            (exp, [1], True)                                {}
placeholder    arg0_1              arg0_1                          ()                                              {}
call_function  broadcast_in_dim_1  prims.broadcast_in_dim.default  (sum_1, [arg0_1, arg0_1, arg0_1], [0, 1, 2])    {}
call_function  div_1               prims.div.default               (exp, broadcast_in_dim_1)                       {}
call_function  permute             aten.permute.default            (div_1, [0, 2, 1])                              {}
call_function  broadcast_in_dim_2  prims.broadcast_in_dim.default  (div_1, [arg0_1, arg0_1, arg0_1], [0, 1, 2])    {}
call_function  broadcast_in_dim_3  prims.broadcast_in_dim.default  (permute, [arg0_1, arg0_1, arg0_1], [0, 1, 2])  {}
call_function  view                aten.view.default               (broadcast_in_dim_2, [arg0_1, arg0_1, arg0_1])  {}
call_function  view_1              aten.view.default               (broadcast_in_dim_3, [arg0_1, arg0_1, arg0_1])  {}
call_function  bmm                 aten.bmm.default                (view, view_1)                                  {}
call_function  view_2              aten.view.default               (bmm, [arg0_1, arg0_1, arg0_1])                 {}
call_function  add_1               prims.add.default               (view_2, view_2)                                {}
call_function  sub_2               prims.sub.default               (add_1, 3.0)                                    {}
call_function  div_2               prims.div.default               (sub_2, 2.0)                                    {}
output         output              output                          (div_2,)                                        {}

// 子图2
opcode         name              target                          args                                         kwargs
-------------  ----------------  ------------------------------  -------------------------------------------  --------
placeholder    arg1_1            arg1_1                          ()                                           {}
call_function  add               prims.add.default               (arg1_1, arg1_1)                             {}
call_function  sub               prims.sub.default               (add, 3.0)                                   {}
call_function  div               prims.div.default               (sub, 2.0)                                   {}
call_function  amax              aten.amax.default               (div, [1], True)                             {}
placeholder    arg0_1            arg0_1                          ()                                           {}
call_function  broadcast_in_dim  prims.broadcast_in_dim.default  (amax, [arg0_1, arg0_1, arg0_1], [0, 1, 2])  {}
call_function  sub_1             prims.sub.default               (div, broadcast_in_dim)                      {}
output         output            output                          (sub_1,)                                     {}
```

编译器会逐一编译这些子图，在torch框架中会有tensor输入，也有符号输入，如果融合的子图比较特殊，仅仅利用输入tensor的信息是无法描述整张图的变换情况，因此必需将这些符号输入传到编译中，于是在之前的基础上做了一些改动，花了挺长一段时间修改以前翻译计算图的逻辑，添加了整图切分的功能，同时也删除了多余的torch symbol op，于是现在编译器的IR图变得比以前简洁了不少啦。

下面展示以下子图：
子图1：

```

module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>, builtin.mode = #llh.Mode<inference>} {
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @main(%arg0: i64 {func.symbol_int = @s0}, %arg1: tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s0"}) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>> attributes {entrance, symbol_int_arg_nums = 1 : i64} {
    %0 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>
    %1 = "llh.constant"() <{value = dense<3.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>
    %2 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %5 = "llh.add"(%arg1, %arg1) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %6 = "llh.dim"(%5, %4) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %7 = "llh.dim"(%5, %3) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %8 = "llh.dim"(%5, %2) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %9 = "llh.broadcast_to"(%1, %6, %7, %8) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64: 0, 1, 2>, noexpand_dims = array<i64>}> : (tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %10 = "llh.sub"(%5, %9) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %11 = "llh.dim"(%10, %4) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %12 = "llh.dim"(%10, %3) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %13 = "llh.dim"(%10, %2) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %14 = "llh.broadcast_to"(%0, %11, %12, %13) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64: 0, 1, 2>, noexpand_dims = array<i64>}> : (tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %15 = "llh.div"(%10, %14) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %16 = "llh.reduce_max"(%15) <{axis = array<i64: 1>}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?xf32, #llh.encoding<shapes = @s0, @s0>>
    %17 = "llh.dim"(%15, %4) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %18 = "llh.dim"(%15, %2) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %19 = "llh.reshape"(%16, %17, %3, %18) : (tensor<?x?xf32, #llh.encoding<shapes = @s0, @s0>>, i64, i64, i64) -> tensor<?x1x?xf32, #llh.encoding<shapes = @s0, @c1, @s0>>
    %20 = "llh.broadcast_to"(%19, %arg0, %arg0, %arg0) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64: 1>, noexpand_dims = array<i64: 0, 2>}> : (tensor<?x1x?xf32, #llh.encoding<shapes = @s0, @c1, @s0>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %21 = "llh.sub"(%15, %20) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    return %21 : tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
  }
}
```

子图2：

```
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>, builtin.mode = #llh.Mode<inference>} {
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @main(%arg0: i64 {func.symbol_int = @s0}, %arg1: tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s0"}) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>> attributes {entrance, symbol_int_arg_nums = 1 : i64} {
    %0 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>
    %1 = "llh.constant"() <{value = dense<3.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>
    %2 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %5 = "llh.reduce_sum"(%arg1) <{axis = array<i64: 1>}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?xf32, #llh.encoding<shapes = @s0, @s0>>
    %6 = "llh.dim"(%arg1, %4) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %7 = "llh.dim"(%arg1, %2) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %8 = "llh.reshape"(%5, %6, %3, %7) : (tensor<?x?xf32, #llh.encoding<shapes = @s0, @s0>>, i64, i64, i64) -> tensor<?x1x?xf32, #llh.encoding<shapes = @s0, @c1, @s0>>
    %9 = "llh.broadcast_to"(%8, %arg0, %arg0, %arg0) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64: 1>, noexpand_dims = array<i64: 0, 2>}> : (tensor<?x1x?xf32, #llh.encoding<shapes = @s0, @c1, @s0>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %10 = "llh.div"(%arg1, %9) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %11 = "llh.transpose"(%10) <{perms = array<i64: 0, 2, 1>}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %12 = "llh.broadcast_to"(%10, %arg0, %arg0, %arg0) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64>, noexpand_dims = array<i64: 0, 1, 2>}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %13 = "llh.broadcast_to"(%11, %arg0, %arg0, %arg0) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64>, noexpand_dims = array<i64: 0, 1, 2>}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %14 = "llh.batch_matmul"(%12, %13) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %15 = "llh.add"(%14, %14) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %16 = "llh.dim"(%15, %4) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %17 = "llh.dim"(%15, %3) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %18 = "llh.dim"(%15, %2) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %19 = "llh.broadcast_to"(%1, %16, %17, %18) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64: 0, 1, 2>, noexpand_dims = array<i64>}> : (tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %20 = "llh.sub"(%15, %19) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %21 = "llh.dim"(%20, %4) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %22 = "llh.dim"(%20, %3) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %23 = "llh.dim"(%20, %2) <{symbol = @s0}> : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, i64) -> i64
    %24 = "llh.broadcast_to"(%0, %21, %22, %23) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64: 0, 1, 2>, noexpand_dims = array<i64>}> : (tensor<1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1>>, i64, i64, i64) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    %25 = "llh.div"(%20, %24) : (tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>, tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>) -> tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
    return %25 : tensor<?x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s0>>
  }
```

好吧，最近一段时间的改动就是这样，emm 其实我一直知道的，我不是很喜欢讲细节的人，所以经常有人对我说的话一头雾水，也有收到一些朋友们的提问，emm如果你们有什么疑问什么的啦可以直接问我吧。

最近工作比较忙，而且遇到一些其他的事情，好像很久没有写记录了，本来想抽空讲一讲MLIR社区的dialect，但是实际感觉也没什么好说的，之后一段时间我要重新设计一下执行器，分阶段整理编译当中的内容，这样不仅调试方便，也可以将整个编译工作分层，好吧不多说了，以后再见吧。

而且这个符号方案花了太多时间做，虽然它有很多潜力还没有发掘出来，但是毕竟不是编译器优化的主要思路，我也觉得是该停一停这方面的精力投入了。
