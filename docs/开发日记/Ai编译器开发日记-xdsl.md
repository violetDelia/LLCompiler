# xDSl

* xDSL简介:https://arxiv.org/pdf/2311.07422
* xDSL简介:https://xdsl.dev/
* MLIR python binding:https://mlir.llvm.org/docs/Bindings/Python/
* MLIR Op定义：[Defining Dialects - MLIR (llvm.org)](https://mlir.llvm.org/docs/DefiningDialects/)

## 为什么选用xDSL

    要从python框架与MLIR上进行对接, 在python上构建Op, 在xDSL出现之前主流的方式是使用MLIR pybinding, 笔者刚开始也想采用这种方式，但是这样构建的话需要导出自定义Dialcet的C语言API，而且容易出现环境问题，笔者刚开始就采用了这种方式，发现运行环境无法正确识别.pyi的文件，过于繁琐的配置方式和环境问题让笔者放弃。
    因此采用xDSL来承载从框架接入MLIR的缓冲层。在Python端生成xDSL的module，将其以字符串的形式传入MLIR中，xDSL的语法简单，应用方便，但是它其实和MLIR是两套系统。这种方式需要在MLIR和xDSL各自定义相同的一组opset，xDSL上的定义可以因为其运行效率不高，可以定义简单一点。在MLIR中定义完整一点。

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

    上图是采用xDSl定义的一个Aot算子，每一个定义的Op都需要加入@irdl_op_definition装饰器，表示其是一个定义的Op，name字段llh.aot的前缀llh表示算子所属的Dialect的名称，aot为算子的名字。

    X = operand_def(TensorType) 表示其为一个变量名为X的操作数，后面括号中TensorType表示该操作数的类型为Tensor。LLH_Computable_Type 定义了一种类型约束，代表该类型可以是Tensor，也可以是整数或者浮点数。熟悉MLIR框架的话很轻易就能看懂，我就不多赘述了。

    在创建该Op时，如同MLIR一样会根据定义进行Op的合法性检测。但是xDSL检测运行缓慢，因此不建议定义定义Op是太过复杂。在接入MLIR时也会进行合法性检测。

## xDSL的弊端
