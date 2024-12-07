# 符号表达方案

整理一下符号表达的思路。

## torch 符号信息传入：

两种方式，
第一个是是从fx_graph上根据faketensor添加函数参数的属性,将torch的符号信息带下来，然后通过符号推导将整图的符号信息推导出来。
第二个是创建torch_symbolic_int 和 symbolic_bind Op，根据这两个Op的信息，生成整图的符号信息。

从torch框架传入入口信息：

1. 根据输入的fx_graph的 faketensor信息，添加参数属性：

```mlir
func.func @main(%arg0: tensor<1x3x?x?xf32> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"})
```

2. 定义torch_symbolic_int 和 symbolic_bind 两个Op，可以从fx_graph 之间创建输出的符号信息（可选）

```mlir
def LLH_TorchSymbolicIntOp : LLH_Op<"torch_symbolic_int",[]>{
    let description = [{
        torch 框架从fx_graph直接获取的符号信息。
    }];
    let arguments = (ins 
       LLH_StringAttr:$sym_name);
    let results = (outs LLH_Int64);
}

def LLH_SymbolicBindOp: Op<LLH_Dialect,"symbolic_bind">{
    let description = [{
        根据fake tensor 的信息,每创建一个op,就相应创建一个symbolic_bind将torch_symbolic_int与Op的结果绑定。
    }];
    let arguments = (ins 
       LLH_Symbolic_Type:$operand ,
       Variadic<LLH_Int64>:$bind_symbols,
       AffineMapAttr:$expressions);
}
```

3. torch_symbolic_int 创建（可选）

```python
if node.op == "placeholder":
    # 张量输入
    if node.type is torch.Tensor:
                pass
    elif node.type is None:
        val = node.meta["val"]
        # 张量输入
        if isinstance(val, FakeTensor):
                    pass
        # 符号输入
        elif isinstance(val, torch.SymInt):
            # placeholder 如果是符号的话，直接在图中创建一个TorchSymbolicIntOp
            op: TorchSymbolicIntOp = torch_symbol_translate(
                        node.meta["val"], symbol_map
                    )
            value_map[node.name] = op.results
            block.add_op(op)
```

4. symbolic_bind 创建（可选）

每创建一个Op,相应为该Op创建一个symbolic_bind Op,创建Op的接口：

```python
def torch_symbol_bind(
    operand: SSAValue, # 创建的Op
    tensor: FakeTensor, # 绑定shape信息的FakeTensor
    symbol_map: dict[str, TorchSymbolicIntOp] #全局符号表【符号名，torch_symbolic_int op】
):
    bind_symbols: list[SSAValue] = []
    affine_expr_map: dict[str, AffineSymExpr] = dict()
    results: list[AffineSymExpr] = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            results.append(AffineConstantExpr(dim))
            continue
        elif str(dim).isdigit():
            results.append(AffineConstantExpr(int(dim)))
            continue
        else:
            affine_exp = _generate_affine_symbolic(
                dim.node.expr, symbol_map, affine_expr_map, bind_symbols
            )
            results.append(affine_exp)
    map = AffineMap(0, len(symbol_map), results=results)
    expressions = AffineMapAttr(map)
    return SymbolicBindOp(
        operands=[operand, bind_symbols], attributes={"expressions": expressions}
    )
```

5. 模型resnet18示例

```mlir
#map = affine_map<()[s0] -> (1, 3, s0, s0)>
#map1 = affine_map<()[s0] -> (1, 64, (s0 - 1) floordiv 2 + 1, (s0 - 1) floordiv 2 + 1)>
#map2 = affine_map<()[s0] -> (1, 64, (s0 - 1) floordiv 4 + 1, (s0 - 1) floordiv 4 + 1)>
#map3 = affine_map<()[s0] -> (1, 128, (s0 - 1) floordiv 8 + 1, (s0 - 1) floordiv 8 + 1)>
#map4 = affine_map<()[s0] -> (1, (((s0 - 1) floordiv 8) * ((s0 - 1) floordiv 8)) * 128 + 128)>
#map5 = affine_map<()[s0] -> (1, 512)>
#map6 = affine_map<()[s0] -> (1, 10)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<1x3x?x?xf32> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<1x10xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llh.weight"() <{weight_file = "xxx.npy"}> : () -> tensor<64x3x7x7xf32>
    ```......```
    %64 = "llh.weight"() <{weight_file = "xxx.npy"}> : () -> tensor<1xi64>
    %65 = "llh.torch_symbolic_int"() <{sym_name = "s1"}> : () -> i64
    "llh.symbolic_bind"(%arg0, %65) <{expressions = #map}> : (tensor<1x3x?x?xf32>, i64) -> ()
    %66 = "llh.conv"(%arg0, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<1x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%66, %65) <{expressions = #map1}> : (tensor<1x64x?x?xf32>, i64) -> ()
    %67 = "llh.batch_norm"(%66, %2, %3, %35, %36) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%67, %65) <{expressions = #map1}> : (tensor<1x64x?x?xf32>, i64) -> ()
    %68 = "llh.relu"(%67) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%68, %65) <{expressions = #map1}> : (tensor<1x64x?x?xf32>, i64) -> ()
    %69 = "llh.max_pool"(%68) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%69, %65) <{expressions = #map2}> : (tensor<1x64x?x?xf32>, i64) -> ()
    %70 = "llh.conv"(%69, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%70, %65) <{expressions = #map2}> : (tensor<1x64x?x?xf32>, i64) -> ()
    %71 = "llh.batch_norm"(%70, %5, %6, %38, %39) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%71, %65) <{expressions = #map2}> : (tensor<1x64x?x?xf32>, i64) -> ()
    %72 = "llh.relu"(%71) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    "llh.symbolic_bind"(%72, %65) <{expressions = #map2}> : (tensor<1x64x?x?xf32>, i64) -> ()
    ```......````
    %107 = "llh.add"(%106, %34) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    "llh.symbolic_bind"(%107) <{expressions = #map6}> : (tensor<1x10xf32>) -> ()
    return %107 : tensor<1x10xf32>
  }
}
```

## 符号推导

1. 定义tensor的Encoding属性：

```mlir
def LLH_Encoding : LLH_Attr<"Encoding", "encoding", []> {
  let description = [{
  "自定义的encoding属性，除了可以存放信息，还可以存放如max_dim、min_dim、layout等tensor的信息 "
  }];
  let parameters = (ins ArrayRefParameter<"::mlir::FlatSymbolRefAttr", "">:$shape_symbols);
  let builders = [
    AttrBuilder<(ins "::mlir::ArrayRef<::mlir::StringRef>":$shape_symbols), [{
      mlir::SmallVector<mlir::FlatSymbolRefAttr> symbols;
      for (auto sym : shape_symbols) {
        symbols.push_back(
            FlatSymbolRefAttr::get($_ctxt, StringAttr::get($_ctxt, sym)));
      }
      return $_get($_ctxt, symbols);
    }]>,
  ];
  //let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    `<`
      `shapes` `=` $shape_symbols
    `>`
  }];
}
```

2. 定义符号推导的接口

```mlir
def SymbolicInferShapeOpInterface: OpInterface<"SymbolicInferShapeOpInterface"> {
  let description = [{
    符号shepa推导的接口.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    /*retTy=*/"::llvm::LogicalResult",/*methodName=*/ "inferSymbolicShape">
  ];
}
```

3. 为每个符号实现符号推导的功能：
   注：    接口不会暴露出来，由全局的SymbolAnalysis来生成符号相关的Op和记录符号之间的关系。同时可以通过SymbolAnalysis来查询符号之间的关系。
   在生成Op的时候会自动推导符号信息。写Pattern时候不需要关注任何符号的信息，开发Pattern可以不感知符号的信息。
   推导实现示例：

```c++
INFER_FUNCTION(MatMulOp) {
  HAS_ENCODING_RETURN(getResult()) // 如果已经生成了Encoding属性，说明该Op的符号信息已经推导过了。退出推导。
  NO_ENCODING_RETURN(getLhs()) // 输入没有Encoding属性，退出推导
  NO_ENCODING_RETURN(getRhs()) // 输入没有Encoding属性，退出推导
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation()); // 获取全局符号分析实例
  // 生成返回值的符号信息
  auto symbols = llvm::SmallVector<StringRef>();  
  auto lhs_type = llc::getRankTensorFrom(getLhs());
  auto rhs_type = llc::getRankTensorFrom(getRhs());
  auto lhs_symbols = llc::getEncodingFrom(lhs_type).getShapeSymbols();
  auto rhs_symbols = llc::getEncodingFrom(rhs_type).getShapeSymbols();
  symbols.push_back(lhs_symbols[0].getValue());
  symbols.push_back(rhs_symbols[1].getValue());

  auto res = getResult();
  symbol_analsis->addEncoding(res, symbols); // 将符号信息附加到Op上
  COMMON_CHECK //一些特殊情况的处理

  // 创建符号之间的关系，matmul的lhs dim1 和 rhs dim0 符号相等
  symbol_analsis->buildSymbolRelation(lhs_symbols[1].getAttr().strref(),
                                      rhs_symbols[0].getAttr().strref(),
                                      SymbolRelation::EQ);
  return llvm::success();
}

```mlir
//推导前
func.func @matmul(%arg0: tensor<?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}) -> tensor<*xf32> attributes {entrance} {
    %const = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<512x10xf32>
    %matmul = "llh.matmul"(%arg0, %const) : (tensor<?x?xf32>, tensor<512x10xf32>) -> tensor<*xf32>
    return %matmul : tensor<*xf32>
}
//推导后发现符号s1的值为512
module {
  "llh.symbolic_int"() <{sym_name = "c10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @matmul(%arg0: tensor<?x?xf32, #llh.encoding<shapes = @s0, @s1>> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>> attributes {entrance} {
    %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<512x10xf32, #llh.encoding<shapes = @c512, @c10>>
    %1 = "llh.matmul"(%arg0, %0) : (tensor<?x?xf32, #llh.encoding<shapes = @s0, @s1>>, tensor<512x10xf32, #llh.encoding<shapes = @c512, @c10>>) -> tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    return %1 : tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
  }
  module @__symbol__ {
    "llh.symbol_relation"() <{relation = @c512, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s1}> : () -> ()
  }
}
```

## 符号优化

1. 定义符号关系的Op

```mlir
def LLH_SymbolBinaryRelationOp : LLH_Op<"symbol_binary_relation",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let description = [{
        描述符号关系的op
    }];
    let arguments = (ins 
       FlatSymbolRefAttr:$symbol,
       FlatSymbolRefAttr:$relations_lhs,
       FlatSymbolRefAttr:$relations_rhs,
       LLH_SymbolRelationsAttr:$relation_kind
       );
    let hasCanonicalizer = 1;
}

def LLH_SymbolRelationOp: LLH_Op<"symbol_relation",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let description = [{
        描述符号关系的op
    }];
    let arguments = (ins 
       FlatSymbolRefAttr:$symbol,
       FlatSymbolRefAttr:$relation,
       LLH_SymbolRelationsAttr:$relation_kind
       );
    let hasCanonicalizer = 1;
}
```

2. 定义符号的规范化方法：

```c++
void SymbolRelationOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {
  results.add<ReplaceSymbolIfEquel>(context);
  results.add<RemoveSymbolRelationIfAllConst>(context);
}
```

3. 优化示例：

```mlir
// 推导出符号相等，进行全局替换：
func.func @relation_eq(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) ->(i64) attributes {entrance} {
  %2 = "llh.constant"() <{symbol = @c0, value = 2 : i64}> : () -> i64
  %0 = "llh.add"(%arg0, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
  %1 = "llh.add"(%arg1, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>
  "llh.encoding_bind"(%1) <{encoding = #llh.encoding<shapes = @s3, @c64, @s4, @s5>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> ()
  %193 = "llh.dim"(%1, %2) <{symbol = @s4}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, i64) -> i64
  return %193: i64
}
module @__symbol__ {
  "llh.symbol_relation"() <{relation = @s5, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s2}> : () -> ()
  "llh.symbol_relation"() <{relation = @s4, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s1}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
}

// 优化后
module {
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @relation_eq(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> i64 attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c0, value = 2 : i64}> : () -> i64
    %1 = "llh.add"(%arg0, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
    %2 = "llh.add"(%arg1, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
    "llh.encoding_bind"(%2) <{encoding = #llh.encoding<shapes = @s0, @c64, @s1, @s2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> ()
    %3 = "llh.dim"(%2, %0) <{symbol = @s1}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, i64) -> i64
    return %3 : i64
  }
}
```

## resnet推导示例：

推导前：

```mlir
func.func @main(%arg0: tensor<1x3x?x?xf32> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<1x10xf32> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %4 = "llh.weight"() <{weight_file = "xxx.npy"}> : () -> tensor<64x3x7x7xf32>
    ```......```
    %67 = "llh.weight"() <{weight_file = "xxx.npy"}> : () -> tensor<1xi64>
    %68 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>>, tensor<64x3x7x7xf32>) -> tensor<1x64x?x?xf32>
    %69 = "llh.batch_norm"(%68, %5, %6, %38, %39) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %70 = "llh.relu"(%69) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %71 = "llh.max_pool"(%70) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %72 = "llh.conv"(%71, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %73 = "llh.batch_norm"(%72, %8, %9, %41, %42) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %74 = "llh.relu"(%73) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %75 = "llh.conv"(%74, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %76 = "llh.batch_norm"(%75, %11, %12, %44, %45) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %77 = "llh.add"(%76, %71) : (tensor<1x64x?x?xf32>, tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %78 = "llh.relu"(%77) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %79 = "llh.conv"(%78, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %80 = "llh.batch_norm"(%79, %14, %15, %47, %48) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %81 = "llh.relu"(%80) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %82 = "llh.conv"(%81, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x?x?xf32>
    %83 = "llh.batch_norm"(%82, %17, %18, %50, %51) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    %84 = "llh.add"(%83, %78) : (tensor<1x64x?x?xf32>, tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %85 = "llh.relu"(%84) : (tensor<1x64x?x?xf32>) -> tensor<1x64x?x?xf32>
    %86 = "llh.conv"(%85, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x?x?xf32>
    %87 = "llh.batch_norm"(%86, %20, %21, %53, %54) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %88 = "llh.relu"(%87) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %89 = "llh.conv"(%88, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x?x?xf32>
    %90 = "llh.batch_norm"(%89, %23, %24, %56, %57) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %91 = "llh.conv"(%85, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<1x128x?x?xf32>
    %92 = "llh.batch_norm"(%91, %26, %27, %59, %60) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %93 = "llh.add"(%90, %92) : (tensor<1x128x?x?xf32>, tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %94 = "llh.relu"(%93) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %95 = "llh.conv"(%94, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x?x?xf32>
    %96 = "llh.batch_norm"(%95, %29, %30, %62, %63) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %97 = "llh.relu"(%96) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %98 = "llh.conv"(%97, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x?x?xf32>
    %99 = "llh.batch_norm"(%98, %32, %33, %65, %66) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x?x?xf32>
    %100 = "llh.add"(%99, %94) : (tensor<1x128x?x?xf32>, tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %101 = "llh.relu"(%100) : (tensor<1x128x?x?xf32>) -> tensor<1x128x?x?xf32>
    %102 = "llh.dim"(%101, %1) : (tensor<1x128x?x?xf32>, i64) -> i64
    %103 = "llh.dim"(%101, %3) : (tensor<1x128x?x?xf32>, i64) -> i64
    %104 = "llh.dim"(%101, %2) : (tensor<1x128x?x?xf32>, i64) -> i64
    %105 = "llh.dim"(%101, %0) : (tensor<1x128x?x?xf32>, i64) -> i64
    %106 = "llh.mul"(%103, %104) : (i64, i64) -> i64
    %107 = "llh.mul"(%106, %105) : (i64, i64) -> i64
    %108 = "llh.reshape"(%101, %102, %107) : (tensor<1x128x?x?xf32>, i64, i64) -> tensor<1x?xf32>
    %109 = "llh.transpose"(%34) <{perms = array<i64: 1, 0>}> : (tensor<512x100352xf32>) -> tensor<100352x512xf32>
    %110 = "llh.matmul"(%108, %109) : (tensor<1x?xf32>, tensor<100352x512xf32>) -> tensor<1x512xf32>
    %111 = "llh.add"(%110, %35) : (tensor<1x512xf32>, tensor<512xf32>) -> tensor<1x512xf32>
    %112 = "llh.dim"(%111, %1) : (tensor<1x512xf32>, i64) -> i64
    %113 = "llh.dim"(%111, %3) : (tensor<1x512xf32>, i64) -> i64
    %114 = "llh.reshape"(%111, %112, %113) : (tensor<1x512xf32>, i64, i64) -> tensor<1x512xf32>
    %115 = "llh.transpose"(%36) <{perms = array<i64: 1, 0>}> : (tensor<10x512xf32>) -> tensor<512x10xf32>
    %116 = "llh.matmul"(%114, %115) : (tensor<1x512xf32>, tensor<512x10xf32>) -> tensor<1x10xf32>
    %117 = "llh.add"(%116, %37) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    return %117 : tensor<1x10xf32>
  }

```

推导后
注: 符号默认是从0开始自动生成的。推导后s3、s15、s16等符号消失，说明在推导的时候发现有相同的符号，被消除了。
同时可以看到还保留了两条符号的关系：s24 = s23 * s22 、s23 = s21 * 128； 这些关系会保存在SymbolAnalysis的关系表中。

```mlir
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "s24"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s23"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s22"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s21"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s20"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s19"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s14"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s13"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s12"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s11"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s9"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s6"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s5"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c128"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @main(%arg0: tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>> {func.input_symbol_0 = "c1", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c10, value = 10 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c512, value = 512 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c128, value = 128 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %5 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %6 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>
    ```......```
    %70 = "llh.conv"(%arg0, %6) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x3x?x?xf32, #llh.encoding<shapes = @c1, @c3, @s1, @s1>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s2>>
    %71 = "llh.batch_norm"(%70, %7, %8, %40, %41) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s2>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s2>>
    %72 = "llh.relu"(%71) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s2>>
    %73 = "llh.max_pool"(%72) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s0, @s2>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %74 = "llh.conv"(%73, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s5, @s6>>
    %75 = "llh.batch_norm"(%74, %10, %11, %43, %44) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s5, @s6>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s5, @s6>>
    %76 = "llh.relu"(%75) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s5, @s6>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s5, @s6>>
    %77 = "llh.conv"(%76, %12) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s5, @s6>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %78 = "llh.batch_norm"(%77, %13, %14, %46, %47) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %79 = "llh.add"(%78, %73) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %80 = "llh.relu"(%79) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %81 = "llh.conv"(%80, %15) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s9, @s10>>
    %82 = "llh.batch_norm"(%81, %16, %17, %49, %50) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s9, @s10>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s9, @s10>>
    %83 = "llh.relu"(%82) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s9, @s10>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s9, @s10>>
    %84 = "llh.conv"(%83, %18) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s9, @s10>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %85 = "llh.batch_norm"(%84, %19, %20, %52, %53) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %86 = "llh.add"(%85, %80) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %87 = "llh.relu"(%86) : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>) -> tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>
    %88 = "llh.conv"(%87, %21) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<128x64x3x3xf32, #llh.encoding<shapes = @c128, @c64, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s13, @s14>>
    %89 = "llh.batch_norm"(%88, %22, %23, %55, %56) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s13, @s14>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s13, @s14>>
    %90 = "llh.relu"(%89) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s13, @s14>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s13, @s14>>
    %91 = "llh.conv"(%90, %24) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s13, @s14>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %92 = "llh.batch_norm"(%91, %25, %26, %58, %59) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %93 = "llh.conv"(%87, %27) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x64x?x?xf32, #llh.encoding<shapes = @c1, @c64, @s11, @s12>>, tensor<128x64x1x1xf32, #llh.encoding<shapes = @c128, @c64, @c1, @c1>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %94 = "llh.batch_norm"(%93, %28, %29, %61, %62) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %95 = "llh.add"(%92, %94) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %96 = "llh.relu"(%95) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %97 = "llh.conv"(%96, %30) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s19, @s20>>
    %98 = "llh.batch_norm"(%97, %31, %32, %64, %65) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s19, @s20>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s19, @s20>>
    %99 = "llh.relu"(%98) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s19, @s20>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s19, @s20>>
    %100 = "llh.conv"(%99, %33) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, weight_layout = #llh.Layout<FCHW>}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s19, @s20>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %101 = "llh.batch_norm"(%100, %34, %35, %67, %68) <{epsilon = 1.000000e-05 : f64, feature_index = 1 : i64, momentum = 1.000000e-01 : f64}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %102 = "llh.add"(%101, %96) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %103 = "llh.relu"(%102) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>) -> tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>
    %104 = "llh.dim"(%103, %4) <{symbol = @s21}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, i64) -> i64
    %105 = "llh.dim"(%103, %3) <{symbol = @s22}> : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, i64) -> i64
    %106 = "llh.mul"(%2, %104) <{symbol = @s23}> : (i64, i64) -> i64
    %107 = "llh.mul"(%106, %105) <{symbol = @s24}> : (i64, i64) -> i64
    %108 = "llh.reshape"(%103, %5, %107) : (tensor<1x128x?x?xf32, #llh.encoding<shapes = @c1, @c128, @s21, @s22>>, i64, i64) -> tensor<1x?xf32, #llh.encoding<shapes = @c1, @s24>>
    %109 = "llh.transpose"(%36) <{perms = array<i64: 1, 0>}> : (tensor<512x100352xf32, #llh.encoding<shapes = @c512, @s24>>) -> tensor<100352x512xf32, #llh.encoding<shapes = @s24, @c512>>
    %110 = "llh.matmul"(%108, %109) : (tensor<1x?xf32, #llh.encoding<shapes = @c1, @s24>>, tensor<100352x512xf32, #llh.encoding<shapes = @s24, @c512>>) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %111 = "llh.reshape"(%37, %5, %1) : (tensor<512xf32, #llh.encoding<shapes = @c512>>, i64, i64) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %112 = "llh.add"(%110, %111) : (tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>, tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %113 = "llh.reshape"(%112, %5, %1) : (tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>, i64, i64) -> tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>
    %114 = "llh.transpose"(%38) <{perms = array<i64: 1, 0>}> : (tensor<10x512xf32, #llh.encoding<shapes = @c10, @c512>>) -> tensor<512x10xf32, #llh.encoding<shapes = @c512, @c10>>
    %115 = "llh.matmul"(%113, %114) : (tensor<1x512xf32, #llh.encoding<shapes = @c1, @c512>>, tensor<512x10xf32, #llh.encoding<shapes = @c512, @c10>>) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    %116 = "llh.reshape"(%39, %5, %0) : (tensor<10xf32, #llh.encoding<shapes = @c10>>, i64, i64) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    %117 = "llh.add"(%115, %116) : (tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>, tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>) -> tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
    return %117 : tensor<1x10xf32, #llh.encoding<shapes = @c1, @c10>>
  }
  module @__symbol__ {
    "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @s23, relations_rhs = @s22, symbol = @s24}> : () -> ()
    "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @c128, relations_rhs = @s21, symbol = @s23}> : () -> ()
  }
}
```

## 符号传递

1. 定义在memref dialcet上使用的表示符号关系的Op

```mlir
def LLH_SymbolicCastOp: Op<LLH_Dialect,"symbolic_cast",[DeclareOpInterfaceMethods<CastOpInterface>]>{
    let arguments = (ins 
       LLH_Symbolic_Type:$operand);
    let results = (outs 
        LLH_Tensor);
}
def LLH_EncodingBindOp : Op<LLH_Dialect,"encoding_bind">{
    let description = [{
        将tensor的encoding信息绑到这个Op上,防止用标准的Pass或者第三方库的Pass出现不识别encoding导致的错误。
    }];
    let arguments = (ins
        LLH_Eencoding_Bind_Type:$operand ,
        LLH_Encoding:$encoding
    );
    let hasCanonicalizer = 1;
}
```

2. 实现Pass将Tensor上的encoding 属性取下来

```mlir
func.func @encoding(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>> attributes {entrance} {
    %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
    return %0 : tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
}

module {
  "llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @encoding(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> attributes {entrance} {
    "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    "llh.encoding_bind"(%0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    return %0 : tensor<?x?x?x?xf32>
  }
}

//to arith
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s3"}) -> tensor<?x?x?x?xf32> attributes {entrance} {
    "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    %0 = arith.addf %arg0, %arg0 : tensor<?x?x?x?xf32>
    "llh.encoding_bind"(%0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    return %0 : tensor<?x?x?x?xf32>
  }
}
//to memref
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: memref<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s3"}) -> memref<?x?x?x?xf32> attributes {entrance} {
    "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 <{symbol = @s0}>: memref<?x?x?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 <{symbol = @s1}>: memref<?x?x?x?xf32>
    %dim_1 = memref.dim %arg0, %c2 <{symbol = @s2}>: memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg0, %c3 <{symbol = @s3}>: memref<?x?x?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0, %dim_1, %dim_2) {alignment = 64 : i64} : memref<?x?x?x?xf32>
    "llh.encoding_bind"(%alloc) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<?x?x?x?xf32>) outs(%alloc : memref<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %in : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<?x?x?x?xf32>
  }
}
//to llvm
module attributes {builtin.gloabal_layout = "NCHW"} {
  llvm.func @main(%arg0: !llvm.ptr) attributes {entrance} {
    %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<128xf32>) : vector<128xf32>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.undef : vector<128xi64>
    %3 = llvm.mlir.constant(dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : vector<128xi64>) : vector<128xi64>
    %4 = llvm.mlir.constant(128 : index) : i64
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.getelementptr inbounds %arg0[9] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
    %10 = llvm.getelementptr inbounds %arg0[8] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
    %12 = llvm.getelementptr inbounds %arg0[7] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %13 = llvm.load %12 : !llvm.ptr -> !llvm.ptr
    %14 = llvm.getelementptr inbounds %arg0[6] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %15 = llvm.load %14 : !llvm.ptr -> !llvm.ptr
    %16 = llvm.getelementptr inbounds %arg0[5] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
    %18 = llvm.getelementptr inbounds %arg0[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %19 = llvm.load %18 : !llvm.ptr -> !llvm.ptr
    %20 = llvm.getelementptr inbounds %arg0[3] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %21 = llvm.load %20 : !llvm.ptr -> !llvm.ptr
    %22 = llvm.getelementptr inbounds %arg0[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %23 = llvm.load %22 : !llvm.ptr -> !llvm.ptr
    %24 = llvm.getelementptr inbounds %arg0[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %25 = llvm.load %24 : !llvm.ptr -> !llvm.ptr
    %26 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
    %27 = llvm.getelementptr inbounds %9[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %28 = llvm.load %27 : !llvm.ptr -> i64
    %29 = llvm.getelementptr inbounds %9[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.load %9 : !llvm.ptr -> i64
    %32 = llvm.getelementptr inbounds %11[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %33 = llvm.load %32 : !llvm.ptr -> i64
    %34 = llvm.getelementptr inbounds %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %35 = llvm.load %34 : !llvm.ptr -> i64
    %36 = llvm.getelementptr inbounds %11[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %37 = llvm.load %36 : !llvm.ptr -> i64
    %38 = llvm.load %11 : !llvm.ptr -> i64
    %39 = llvm.load %13 : !llvm.ptr -> i64
    %40 = llvm.getelementptr inbounds %19[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %41 = llvm.load %40 : !llvm.ptr -> i64
    %42 = llvm.getelementptr inbounds %19[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %43 = llvm.load %42 : !llvm.ptr -> i64
    %44 = llvm.load %19 : !llvm.ptr -> i64
    %45 = llvm.getelementptr inbounds %21[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %46 = llvm.load %45 : !llvm.ptr -> i64
    %47 = llvm.getelementptr inbounds %21[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %48 = llvm.load %47 : !llvm.ptr -> i64
    %49 = llvm.getelementptr inbounds %21[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %50 = llvm.load %49 : !llvm.ptr -> i64
    %51 = llvm.load %21 : !llvm.ptr -> i64
    %52 = llvm.load %23 : !llvm.ptr -> i64
    %53 = llvm.insertvalue %17, %5[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %54 = llvm.insertvalue %15, %53[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.insertvalue %39, %54[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %38, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.insertvalue %31, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %37, %57[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.insertvalue %30, %58[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = llvm.insertvalue %35, %59[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.insertvalue %28, %60[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %62 = llvm.insertvalue %33, %61[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    "llh.encoding_bind"(%62) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    %63 = llvm.insertvalue %26, %5[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %64 = llvm.insertvalue %25, %63[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %65 = llvm.insertvalue %52, %64[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %66 = llvm.insertvalue %51, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %67 = llvm.insertvalue %44, %66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.insertvalue %50, %67[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.insertvalue %43, %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %70 = llvm.insertvalue %48, %69[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.insertvalue %41, %70[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.insertvalue %46, %71[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    "llh.encoding_bind"(%72) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
    %73 = llvm.extractvalue %72[3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.alloca %7 x !llvm.array<4 x i64> : (i64) -> !llvm.ptr
    llvm.store %73, %74 : !llvm.array<4 x i64>, !llvm.ptr
    %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %76 = llvm.load %75 : !llvm.ptr -> i64
    %77 = llvm.alloca %7 x !llvm.array<4 x i64> : (i64) -> !llvm.ptr
    llvm.store %73, %77 : !llvm.array<4 x i64>, !llvm.ptr
    %78 = llvm.getelementptr %77[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %79 = llvm.load %78 : !llvm.ptr -> i64
    %80 = llvm.alloca %7 x !llvm.array<4 x i64> : (i64) -> !llvm.ptr
    llvm.store %73, %80 : !llvm.array<4 x i64>, !llvm.ptr
    %81 = llvm.getelementptr %80[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %82 = llvm.load %81 : !llvm.ptr -> i64
    %83 = llvm.alloca %7 x !llvm.array<4 x i64> : (i64) -> !llvm.ptr
    llvm.store %73, %83 : !llvm.array<4 x i64>, !llvm.ptr
    %84 = llvm.getelementptr %83[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %85 = llvm.load %84 : !llvm.ptr -> i64
    llvm.br ^bb1(%6 : i64)
  ^bb1(%86: i64):  // 2 preds: ^bb0, ^bb11
    %87 = llvm.icmp "slt" %86, %76 : i64
    llvm.cond_br %87, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%88: i64):  // 2 preds: ^bb2, ^bb10
    %89 = llvm.icmp "slt" %88, %79 : i64
    llvm.cond_br %89, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%6 : i64)
  ^bb5(%90: i64):  // 2 preds: ^bb4, ^bb9
    %91 = llvm.icmp "slt" %90, %82 : i64
    llvm.cond_br %91, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%6 : i64)
  ^bb7(%92: i64):  // 2 preds: ^bb6, ^bb8
    %93 = llvm.icmp "slt" %92, %85 : i64
    llvm.cond_br %93, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %94 = llvm.alloca %7 x !llvm.array<4 x i64> : (i64) -> !llvm.ptr
    llvm.store %73, %94 : !llvm.array<4 x i64>, !llvm.ptr
    %95 = llvm.getelementptr %94[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %96 = llvm.load %95 : !llvm.ptr -> i64
    %97 = llvm.sub %96, %92 : i64
    %98 = llvm.insertelement %97, %2[%1 : i32] : vector<128xi64>
    %99 = llvm.shufflevector %98, %2 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<128xi64> 
    %100 = llvm.icmp "sgt" %99, %3 : vector<128xi64>
    %101 = llvm.mul %86, %44 : i64
    %102 = llvm.mul %88, %43 : i64
    %103 = llvm.add %101, %102 : i64
    %104 = llvm.mul %90, %41 : i64
    %105 = llvm.add %103, %104 : i64
    %106 = llvm.add %105, %92 : i64
    %107 = llvm.getelementptr %25[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.intr.masked.load %107, %100, %0 {alignment = 4 : i32} : (!llvm.ptr, vector<128xi1>, vector<128xf32>) -> vector<128xf32>
    %109 = llvm.fadd %108, %108  : vector<128xf32>
    %110 = llvm.extractvalue %62[3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %111 = llvm.alloca %7 x !llvm.array<4 x i64> : (i64) -> !llvm.ptr
    llvm.store %110, %111 : !llvm.array<4 x i64>, !llvm.ptr
    %112 = llvm.getelementptr %111[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %113 = llvm.load %112 : !llvm.ptr -> i64
    %114 = llvm.sub %113, %92 : i64
    %115 = llvm.insertelement %114, %2[%1 : i32] : vector<128xi64>
    %116 = llvm.shufflevector %115, %2 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<128xi64> 
    %117 = llvm.icmp "sgt" %116, %3 : vector<128xi64>
    %118 = llvm.mul %86, %31 : i64
    %119 = llvm.mul %88, %30 : i64
    %120 = llvm.add %118, %119 : i64
    %121 = llvm.mul %90, %28 : i64
    %122 = llvm.add %120, %121 : i64
    %123 = llvm.add %122, %92 : i64
    %124 = llvm.getelementptr %15[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.intr.masked.store %109, %124, %117 {alignment = 4 : i32} : vector<128xf32>, vector<128xi1> into !llvm.ptr
    %125 = llvm.add %92, %4 : i64
    llvm.br ^bb7(%125 : i64)
  ^bb9:  // pred: ^bb7
    %126 = llvm.add %90, %7 : i64
    llvm.br ^bb5(%126 : i64)
  ^bb10:  // pred: ^bb5
    %127 = llvm.add %88, %7 : i64
    llvm.br ^bb3(%127 : i64)
  ^bb11:  // pred: ^bb3
    %128 = llvm.add %86, %7 : i64
    llvm.br ^bb1(%128 : i64)
  ^bb12:  // pred: ^bb1
    llvm.return
  }
}

在下降到 llvm 后 去掉llh.encoding_bind op。
```

## 自动推导

重写PatternRewriter,在创建Op时调用processWileBuildOperation(),进行shape和符号的推导。

```c++
class LLHPatternRewriter : public RewriterBase {
 public:
  explicit LLHPatternRewriter(MLIRContext *ctx) : RewriterBase(ctx) {}
  explicit LLHPatternRewriter(Operation *op) : RewriterBase(op) {}
  using RewriterBase::RewriterBase;

  virtual void processWileBuildOperation(Operation *op);

  virtual bool canRecoverFromRewriteFailure() const;

 public:
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    auto op = RewriterBase::create<OpTy>(location, std::forward<Args>(args)...);
    processWileBuildOperation(op);
    return op;
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value>
  createOrFold(Location location, Args &&...args) {
    SmallVector<Value, 1> results;
    RewriterBase::createOrFold<OpTy>(results, location,
                                     std::forward<Args>(args)...);
    auto op = results.front().getDefiningOp();
    processWileBuildOperation(op);
    return results.front();
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy>
  createOrFold(Location location, Args &&...args) {
    auto op = create<OpTy>(location, std::forward<Args>(args)...);
    SmallVector<Value, 0> unused;
    (void)tryFold(op.getOperation(), unused);
    return op;
  }

  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(Operation *op, Args &&...args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOp(op, newOp.getOperation());
    return newOp;
  }
};
```

### 更多Op推导实现

在checkAndInferSymbol(),中，可以定义为实现SymbolicInferShapeOpInterface的Op的推导，比如tensor.dim,memref.alloc 等。

在SymbolAnalysis定义为额外需要推导的Op，通过getOrBuildSymbolAttrFrom 和 getOrBuildEncodingBindFrom 生成必要的符号信息。

```c++
void LLHPatternRewriter::processWileBuildOperation(Operation *op) {
  llh::checkAndInferSymbol(op);
  //llh::checkBroadcast(op);
}

void checkAndInferSymbol(Operation* op) {
  if (!SymbolAnalysis::symbol_enable) return;
  auto symbol_op = llvm::dyn_cast_or_null<SymbolicInferShapeOpInterface>(op);
  if (symbol_op) {
    symbol_op.inferSymbolicShape();
    return;
  }
  if (SymbolAnalysis::isExtraSymbolicInferOp(op)) {
    SymbolAnalysis::getInstance(op)->getOrBuildSymbolAttrFrom(op);
    SymbolAnalysis::getInstance(op)->getOrBuildEncodingBindFrom(op);
  }
}
```

## 使用Tablegen来进行IR的重写

符号系统基本不影响原有MLIR框架的Pass重写和开发。开发不需要考虑Op的符号情况。
如果Pat 中 需要有新符号的生成，则不能使用TableGen。
用tablegen 实现 abs(abs(x)) --> abs(x)
示例：

```mlir

def FoldTwoAbsOpPattern : Pat<(LLH_AbsOp (LLH_AbsOp:$res $input)),
                                 (LLH_AbsOp $input),
                                 [(HasOneUse $res)]>;
```

```c++

void AbsOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldTwoAbsOpPattern>(context);
  results.add<EraseNoUserOp<AbsOp>>(context);
}
```

```mlir
func.func @fold_two_abs(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) ->  tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>> attributes {entrance} {
  %4 = "llh.abs"(%arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
  %5 = "llh.abs"(%4) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
  return %5 : tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
}

// 重写后
func.func @fold_two_abs(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>> attributes {entrance} {
    %0 = "llh.abs"(%arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
    return %0 : tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
  }
```
