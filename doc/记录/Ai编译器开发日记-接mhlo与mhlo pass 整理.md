# mlho
为了方便表达reduce类的算子以及更高效的动态算子代码生成（ps，tosa的代码生成因为其支持广播，所以生成的代码有很多分支判断检测要不要广播，导致不好处理）。将原有的从stablehlo以及Tosa的Lowing迁至mhlo直接到linalg。整理了一下mlho的pass。

## Pass

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

## 


    




