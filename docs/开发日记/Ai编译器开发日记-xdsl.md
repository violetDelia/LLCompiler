# xDSl

* xDSL简介:https://arxiv.org/pdf/2311.07422
* xDSL简介:https://xdsl.dev/
* MLIR python binding:https://mlir.llvm.org/docs/Bindings/Python/
* MLIR Op定义：[Defining Dialects - MLIR (llvm.org)](https://mlir.llvm.org/docs/DefiningDialects/)

## 为什么选用xDSL

    要实现Python前端神经网络框架与MLIR的对接，需要在Python中构建Op。在xDSL出现之前，主流方法是使用MLIR的pybinding。笔者最初考虑采用这种方式，但这需要导出自定义Dialect的C语言API，同时容易引发环境问题。开始时，笔者尝试了这种方法，但发现运行环境无法正确识别.pyi文件，配置过程繁琐且问题频出，最终决定放弃这种方案。
	因此采用xDSL来承载从框架接入MLIR的缓冲层。在Python端生成xDSL的module，将其以字符串的形式传入MLIR中，xDSL的语法简单，应用方便。

## xDSL上定义Op实例

```python
LLH_Computable_Type = ContainerOf(
    AnyOf(
        [TensorType, IntegerType, Float16Type, Float32Type, Float64Type, BFloat16Type]
    )
)

@irdl_op_definition
class AOTOp(IRDLOperation):
    name = "llh.aot"
    name = attr_def(StringAttr)
    inputs = var_operand_def(LLH_Computable_Type)
    outputs = var_result_def(LLH_Computable_Type)

@irdl_op_definition
class ConvOp(IRDLOperation):
    name = "llh.conv"
    X = operand_def(TensorType)
    W = operand_def(TensorType)
    dilation = attr_def(ArrayAttr)
    kernel_shape = attr_def(ArrayAttr)
    pad = attr_def(ArrayAttr)
    stride = attr_def(ArrayAttr)
    group = attr_def(IntegerAttr)
    result = result_def(TensorType)
```

    上图展示了一个采用xDSL定义的AOT算子。每个定义的Op都需要添加`@irdl_op_definition`装饰器，以标明其为一个定义的Op。字段 `name`中的前缀 `llh`表示该算子所属的Dialect名称，而 `aot`则是算子的具体名称。

    `X = operand_def(TensorType)`表示一个名为 `X`的操作数，其类型为 `Tensor`。`LLH_Computable_Type`定义了一种类型约束，表示该类型可以是Tensor、整数或浮点数。对于熟悉MLIR框架的人来说，这些概念非常容易理解，因此不再详细赘述。

    在创建Op时，类似于MLIR，xDSL会根据定义进行合法性检测。然而，由于xDSL的检测速度较慢，因此不建议定义过于复杂的Op。在接入MLIR后，同样会进行合法性检测，以确保操作的正确性。

## xDSL的弊端

    采用xDSL最大的弊端就是运行构图的运行效率，尤其是建立DenseAttr时，会花费大量的时间，为此，笔者定义了一个特殊的Op-WeightOp，这个Op会将模型的权重文件保存在weight_file中，在MLIR里面必要时读取对应的文件内容，获取权重的输出。这样做可以大幅降低xDSL的运行时间。

    另一个好处是可以在编译时将Weight数据不直接编译到模型文件中，这样不仅减少了编译时间和编译后模型文件的大小，还方便更换模型权重。只需修改权重文件的保存路径，而无需重新编译，大大提高了灵活性。

```python
@irdl_op_definition
class WeightOp(IRDLOperation):
    name = "llh.weight"
    weight_file = attr_def(StringAttr)
    result = result_def(TensorType)
```

    另一个不足之处在于，xDSL和MLIR实际上是两套系统。这种方式要求在两者中各自定义相同的一组opset，而由于xDSL的运行效率较低，可以选择定义得简单一些，而在MLIR中则需要更完整的定义，这增加了维护的复杂性。
