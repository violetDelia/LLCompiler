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

| op name           | xdsl | verify  | shape infer | fold   | canonicalization | fusion type | decomposition | quantify | aot |     |     |
| ----------------- | ---- | ------- | ----------- | ------ | ---------------- | ----------- | ------------- | -------- | --- | --- | --- |
| AdaptiveAvgPoolOp | Yes  | default | [need]      | [need] | [need]           | [need]      | [need]        | [need]   |     |     |     |
| AddOp             | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| AOTOp             | Yes  | default | None        | None   | None             | None        | None          | None     |     |     |     |
| BatchNormOp       | Yes  | default | [need]      | [need] | [need]           | [need]      | [need]        | [need]   |     |     |     |
| CatOp             | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| ConstantOp        | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| ConvBiasOp        | Yes  | default | [need]      | [need] | [need]           | [need]      | [need]        | [need]   |     |     |     |
| ConvOp            | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| DimOp             | Yes  | default | None        | None   | None             | None        | None          | None     |     |     |     |
| DivOp             | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| DropOp            | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| ExpandOp          | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| FlattenOp         | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| LayerNormOp       | Yes  | default | [need]      | [need] | [need]           | [need]      | [need]        | [need]   |     |     |     |
| MatmulOp          | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| MaxPoolOp         | Yes  | default | [need]      | [need] | [need]           | [need]      | [need]        | [need]   |     |     |     |
| MulOp             | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| ReluOp            | Yes  | default | [need]      | [need] | [need]           | [need]      | [need]        | [need]   |     |     |     |
| ReshapeOp         | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| shapeOfOp         | Yes  | default | None        | None   | None             | None        | None          | None     |     |     |     |
| SymbolicBindOp    | Yes  | default | None        | None   | None             | None        | None          | None     |     |     |     |
| SymbolicIntOp     | Yes  | default | None        | None   | None             | None        | None          | None     |     |     |     |
| TransposeOp       | Yes  | default | [need]      | [need] | [need]           | [need]      | None          | [need]   |     |     |     |
| WeightOp          | Yes  | default | None        | None   | [need]           | None        | None          | None     |     |     |     |
