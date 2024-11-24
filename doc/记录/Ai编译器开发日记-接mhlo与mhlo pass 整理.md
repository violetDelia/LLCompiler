# MHLO
为了复用mhlo的一些功能，将原来从stablehlo -> tosa 的路线 迁移到了 mhlo ->stablehlo -> tosa ， 本来想直接从mhlo-> linalg的，但是发现tosa上的优化还是有点用的。

整理了一下mhlo主线上的Pass。

## mhlo Pass

- 预处理

  - expand-hlo-tuples 去除tuple，因为llvm.func不容易实现返回tuple
  - convert-to-signless 统一Int类型为signless
  - group-reduction-dimensions 减少reduce 的维度，可以方便做代码生成
  - mhlo-collapse-elementwise-map   将map打散为普通的Op，不然做不了图优化
  - mhlo-test-lower-complex  去除mhlo.complex
  - mhlo-restrict-max-rank 限制张量最大的rank
- 规范化

  - constraint-fusion  shape dialect 的融合。应该在规范化shape dialect 之前。
  - mhlo-merge-assuming-ops  shape Dialect上的融合
  - mhlo-test-materialize-broadcasts  显示添加广播
  - hlo-canonicalize-dot 规范化dot
  - hlo-canonicalize-gather 规范化gather
  - hlo-canonicalize-reduction 规范化reduction
  - hlo-canonicalize-scatter 规范化scatter
- 算子替换/简化

  - mhlo-legalize-create-token-to-after-all  简化create_token
  - mhlo-legalize-broadcast-to-broadcast-in-dim   简化broadcast
  - mhlo-legalize-dot-general-to-dot   相互替换
  - mhlo-legalize-dot-to-dot-general   相互替换
  - mhlo-legalize-einsum-to-dot-general   简化einsum
  - mhlo-legalize-gather-to-torch-index-select  相互替换
  - mhlo-legalize-torch-index-select-to-gather 相互替换
  - hlo-legalize-sort  简化sort到arith Dialect
  - hlo-legalize-to-arithmetic  简化rng_get_and_update_state到arith Dialect
  - mhlo-test-lower-general-dot  简化general-dot
  - mhlo-test-optimize 简化gather
- 优化

  - mhlo-broadcast-propagation 传播broadcast,方便之后做braodcast消除。
  - mhlo-expand-ops-simplifier 拆解复杂的Op(select_and_scatter)为小算子组合
  - mhlo-sink-constants-to-control-flow  常量下沉
  - symbolic-shape-optimization  动态reshape 和 braodcast的优化
  - mhlo-test-unfuse-batch-norm 将batch-norm 拆解为小算子组合
- Lowing

  - shape-legalize-to-hlo   From Shape Dialect to Mhlo Dialect
  - hlo-legalize-to-linalg  to linalg Dialect
  - hlo-legalize-to-memref  to memref Dialect （主要是广播和reshape）
  - hlo-legalize-to-stablehlo   to stablehlo Dialect
  - mhlo-legalize-control-flow  to scf Dialect
  - hlo-legalize-shape-computations  一些跟shape的Oplowing，应该最先做。
  - mhlo-legalize-to-std  简单的elewiseOp 直接到arith
- 其他

  - buffer_reuse 内存复用

## 新的lowing  Pepeline

```
  //===----------------------------------------------------------------------===//
  //  opt mhlo
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createCanonicalizerPass());
  // 简化reshape  braodcast
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSymbolicShapeOptimizationPass());
  // lowing shape 相关的计算
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());
  // 去除tuple
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createFlattenTuplePass());
  // 简化reduce
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createGroupReductionDimensionsPass());
  // 规范braodcast-->broadcast-in-dim
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeBroadcastToBroadcastInDimPass());
  // 规范化 dot
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeDotPass());
  // 规范化reduce
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeReductionPass());
  // 规范化 gather
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeGatherPass());
  // 规范化scatter
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeScatterPass());
  // 移动常量到控制流内部
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSinkConstantsToControlFlowPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // 简化算子
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createMhloExpandOpsSimplifierPass());
  // 简化算子 reduce
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createGroupReductionDimensionsPass());
  // 算子替换
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeDotToDotGeneralPass());
  // 算子拆解BatchNorm
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createTestUnfuseBatchNormPass());
  // 传播广播
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createBroadcastPropagationPass());

  pm.addPass(mlir::createCanonicalizerPass());

  //===----------------------------------------------------------------------===//
  //  lowing mhlo
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeToStdPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  pm.addNestedPass<mlir::ModuleOp>(mlir::mhlo::createLegalizeToMemrefPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeControlFlowPass());

  //===----------------------------------------------------------------------===//
  //   hlo opt
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloAggressiveFolderPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloAggressiveSimplificationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloCompatibilityExpanderPass());  //分解
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloLegalizeDeprecatedOpsPass());
  pm.addPass(mlir::stablehlo::createStablehloConvertToSignlessPass());
  pm.addPass(mlir::createCanonicalizerPass());

  //===----------------------------------------------------------------------===//
  //  lowing hlo
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createStablehloQuantLegalizeToTosaRescalePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createStablehloPrepareForTosaPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createStablehloLegalizeToTosaPass());

  //===----------------------------------------------------------------------===//
  //  tosa opt
  //===----------------------------------------------------------------------===//
  // tosa 常量折叠
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaLayerwiseConstantFoldPass(
          {.aggressiveReduceConstant = true}));
  // 算子分解
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaOptionalDecompositions());
  // transpose 消除
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaReduceTransposes());
  pm.addPass(mlir::createCanonicalizerPass());

  //===----------------------------------------------------------------------===//
  //  lowing tosa
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass(
      {.enablePrimitiveOps = true}));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed(
      {.preferConv2DKernelLayoutHWCF = false}));

  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  pm.addPass(mlir::tosa::createTosaToArith(true));
```

## 一些坑的地方

1. 对动态shape的conv，dot lowing linalg好像不全面。看来只好自己写conv的codegen了。
2. symbolic-shape-optimization 直接将tensor.dim 和 shape.dim 设为非法的op了，但是支持不了所有情况。
3. mhlo OneShotBufferizePass 的和 DropEquivalentBufferResultsPass会有bug，不清楚为什么。
