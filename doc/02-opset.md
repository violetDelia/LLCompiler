# OpSet

- xdsl : xdsl 是否定义
- op name : Op名
- verify : 合法性检验。
- [infer]shape infer : 形状推导
- fold : 常量折叠
- [Can]canonicalization : 规范化
- [FT]fusion type : 融合类型
- [DE]decomposition: 算子拆解
- [Q]quantify: 量化
- aot: 是否支持aot
- decomposition/convert

| op name            | xdsl | verify  | infer  | fold   | Can    | FT     | DE     | Q      | aot    | lowing |     |
| ------------------ | ---- | ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | --- |
| AdaptiveAvgPoolOp  | Yes  | [need]  | [need] | [need] | [need] | [need] | [need] | [need] | [need] | [need] |     |
| AddOp              | Yes  | default | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| AOTOp              | Yes  | default | None   | None   | None   | None   | None   | None   | [need] | [need] |     |
| BatchNormOp        | Yes  | [need]  | Yes    | [need] | [need] | [need] | [need] | [need] | [need] | [need] |     |
| CatOp              | Yes  | [need]  | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| ConstantOp         | Yes  | default | Yes    | Yes    | [need] | [need] | None   | [need] | [need] | [need] |     |
| ConvBiasOp         | Yes  | [need]  | [need] | [need] | [need] | [need] | [need] | [need] | [need] | [need] |     |
| ConvOp             | Yes  | [need]  | Yes    | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| DimOp              | Yes  | default | None   | [need] | None   | None   | None   | None   | [need] | [need] |     |
| DivOp              | Yes  | default | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| DropOp             | Yes  | default | Yes    | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| EmptyOp            | Yes  | default | Yes    | None   | None   | None   | None   | None   | None   | [need] |     |
| ExpandOp           | Yes  | [need]  | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| FlattenOp          | Yes  | default | None   | None   | None   | None   | Yes    | None   | None   | None   |     |
| LayerNormOp        | Yes  | [need]  | [need] | [need] | [need] | [need] | [need] | [need] | [need] | None   |     |
| MatmulOp           | Yes  | [need]  | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| MaxPoolOp          | Yes  | [need]  | [need] | [need] | [need] | [need] | [need] | [need] | [need] | [need] |     |
| MulOp              | Yes  | default | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| ReluOp             | Yes  | default | Yes    | [need] | [need] | [need] | [need] | [need] | [need] | [need] |     |
| ReshapeOp          | Yes  | default | Yes    | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| shapeOfOp          | Yes  | default | None   | None   | [need] | None   | None   | None   | None   | None   |     |
| SubOp              | Yes  | default | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| SymbolicBindOp     | Yes  | default | None   | None   | None   | None   | None   | None   | None   | NOne   |     |
| SymbolicIntOp      | Yes  | default | None   | None   | [need] | None   | None   | None   | None   | None   |     |
| TorchSymbolicIntOp | Yes  | default | None   | None   | None   | None   | None   | None   | None   | None   |     |
| TransposeOp        | Yes  | default | [need] | [need] | [need] | [need] | None   | [need] | [need] | [need] |     |
| WeightOp           | Yes  | default | Yes    | None   | Yes    | None   | None   | None   | None   | None   |     |

