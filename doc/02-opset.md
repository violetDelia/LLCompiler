# OpSet

- xdsl : xdsl 是否定义
- op name : Op名
- verify : 合法性检验。
- shape infer : 形状推导
- fold : 常量折叠
- canonicalization : 规范化
- fusion type : 融合类型
- decomposition: 算子拆解
- quantify: 量化
- aot: 是否支持aot

| op name            | xdsl | verify | shape infer | fold   | canonicalization | fusion type | decomposition | quantify | aot    | 备注 |     |
| ------------------ | ---- | ------ | ----------- | ------ | ---------------- | ----------- | ------------- | -------- | ------ | ---- | --- |
| AdaptiveAvgPoolOp  | Yes  | [need] | [need]      | [need] | [need]           | [need]      | [need]        | [need]   | [need] |      |     |
| AddOp              | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| AOTOp              | Yes  | [need] | None        | None   | None             | None        | None          | None     | [need] |      |     |
| BatchNormOp        | Yes  | [need] | [need]      | [need] | [need]           | [need]      | [need]        | [need]   | [need] |      |     |
| CatOp              | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| ConstantOp         | Yes  | [need] | Yes    | Yes    | [need]           | [need]      | None          | [need]   | [need] |      |     |
| ConvBiasOp         | Yes  | [need] | [need]      | [need] | [need]           | [need]      | [need]        | [need]   | [need] |      |     |
| ConvOp             | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| DimOp              | Yes  | [need] | None        | None   | None             | None        | None          | None     | [need] |      |     |
| DivOp              | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| DropOp             | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| ExpandOp           | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| FlattenOp          | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| LayerNormOp        | Yes  | [need] | [need]      | [need] | [need]           | [need]      | [need]        | [need]   | [need] |      |     |
| MatmulOp           | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| MaxPoolOp          | Yes  | [need] | [need]      | [need] | [need]           | [need]      | [need]        | [need]   | [need] |      |     |
| MulOp              | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| ReluOp             | Yes  | [need] | [need]      | [need] | [need]           | [need]      | [need]        | [need]   | [need] |      |     |
| ReshapeOp          | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| shapeOfOp          | Yes  | [need] | None        | None   | None             | None        | None          | None     | None   |      |     |
| SymbolicBindOp     | Yes  | [need] | None        | None   | None             | None        | None          | None     | None   |      |     |
| SymbolicIntOp      | Yes  | [need] | None        | None   | None             | None        | None          | None     | None   |      |     |
| TorchSymbolicIntOp | Yes  | Yes    | None        | None   | None             | None        | None          | None     | None   |      |     |
| TransposeOp        | Yes  | [need] | [need]      | [need] | [need]           | [need]      | None          | [need]   | [need] |      |     |
| WeightOp           | Yes  | [need] | Yes        | None   | [need]           | None        | None          | None     | None   |      |     |
