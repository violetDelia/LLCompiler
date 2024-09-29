Pass Manager with 6 passes:
builtin.module(operation-legalization,inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },infer-symbol-shape,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},load-weight,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})

[2024-09-30 00:14:33.979] [info] ----- run in pass: Operationlegalization -----
[2024-09-30 00:14:33.979] [info] ----- run out pass: Operationlegalization -----
[2024-09-30 00:14:33.981] [info] ----- run in pass: InferSymbolShape -----
[2024-09-30 00:14:33.981] [info] c3
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
[2024-09-30 00:14:33.981] [info] s0
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
[2024-09-30 00:14:33.981] [info] s1
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
[2024-09-30 00:14:33.982] [info] s2
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.constant
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.constant
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.constant
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.constant
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.weight
#llh.encoding<shapes = @s0, @c3, @s1, @s2>
#llh.encoding<shapes = @c64, @c3, @c7, @c7>
<block argument> of type 'tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>' at index: 0
array<i64: 7, 7>
array<i64: 3, 3, 3, 3>
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.max_pool
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.conv
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.batch_norm
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.relu
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.adaptive_average_pool
[2024-09-30 00:14:33.982] [warning] /home/lfr/LLCompiler/src/Dialect/LLH/IR/LLHinfersymbolShape.cpp<99>: 
	function [inferSymbolicShape] Unimplemented! op name:llh.div
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.div
[2024-09-30 00:14:33.982] [warning] /home/lfr/LLCompiler/src/Dialect/LLH/IR/LLHinfersymbolShape.cpp<99>: 
	function [inferSymbolicShape] Unimplemented! op name:llh.div
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.div
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.reshape
#llh.encoding<shapes = @c1000, @c512>
[2024-09-30 00:14:33.982] [warning] /home/lfr/LLCompiler/src/Dialect/LLH/IR/LLHinfersymbolShape.cpp<103>: 
	function [inferSymbolicShape] Unimplemented! op name:llh.transpose
[2024-09-30 00:14:33.982] [info] Inferred symbolic shapellh.transpose
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.matmul
<<NULL ATTRIBUTE>>
[2024-09-30 00:14:33.982] [error] Invalid operand to infer symbolllh.add
[2024-09-30 00:14:33.983] [info] ----- run out pass: InferSymbolShape -----
[2024-09-30 00:14:33.984] [info] ----- run in pass: LoadWeight -----
[2024-09-30 00:14:33.984] [error] fp : /home/lfr/LLCompiler/src/Dialect/LLH/Transforms/LoadWeight.cpp<115> 
	read file error: /home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-28T17:08:16.961145+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.	Program arguments: /home/lfr/LLCompiler/build/bin/llc-opt --dump-pass-pipeline -o=/home/lfr/LLCompiler/out.mlir --log-lever=debug --log-root=C:codingLLCompilerlog --mlir-print-ir-tree-dir=/home/lfr/LLCompiler/ir_tree --mlir-print-ir-after-all -basic-pipeline /home/lfr/LLCompiler/test/model_ir/resnet18.mlir
Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
0  libLLVMSupport.so.20.0git        0x00007f38428dd800 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) + 240
1  libLLVMSupport.so.20.0git        0x00007f38428dad6a llvm::sys::RunSignalHandlers() + 58
2  libLLVMSupport.so.20.0git        0x00007f38428dafd5
3  libc.so.6                        0x00007f38422b3520
4  libc.so.6                        0x00007f38422f0b52 fread + 34
5  libMLIRLLHTransforms.so          0x00007f38443c7e93
6  libMLIRLLHTransforms.so          0x00007f38443c98dc
7  libMLIRLLHTransforms.so          0x00007f38443dd884 mlir::detail::LLCOpOrInterfaceRewritePatternBase<mlir::llh::WeightOp>::matchAndRewrite(mlir::llh::WeightOp, mlir::LLCPatternRewriter&) const + 68
8  libMLIRLLHTransforms.so          0x00007f38443ca1a9 mlir::detail::LLCOpOrInterfaceRewritePatternBase<mlir::llh::WeightOp>::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const + 89
9  libMLIRRewrite.so.20.0git        0x00007f384146d908 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) + 2776
10 libMLIRTransformUtils.so.20.0git 0x00007f38414e7cce
11 libMLIRTransformUtils.so.20.0git 0x00007f38414ea4d5 mlir::applyPatternsAndFoldGreedily(mlir::Region&, mlir::FrozenRewritePatternSet const&, mlir::GreedyRewriteConfig, bool*) + 1093
12 libMLIRLLHTransforms.so          0x00007f38443c7612
13 libMLIRPass.so.20.0git           0x00007f3842c523d9 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) + 1193
14 libMLIRPass.so.20.0git           0x00007f3842c52891 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) + 321
15 libMLIRPass.so.20.0git           0x00007f3842c538e5 mlir::PassManager::run(mlir::Operation*) + 1445
16 libMLIROptLib.so.20.0git         0x00007f384445a2b7
17 libMLIROptLib.so.20.0git         0x00007f384445accc
18 libMLIROptLib.so.20.0git         0x00007f384445ae2d
19 libMLIRSupport.so.20.0git        0x00007f3841d148de mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) + 174
20 libMLIROptLib.so.20.0git         0x00007f38444514bc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) + 220
21 libMLIROptLib.so.20.0git         0x00007f384445af90 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) + 304
22 libMLIROptLib.so.20.0git         0x00007f384445b4b7 mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) + 391
23 llc-opt                          0x0000563a3ced9a7b
24 libc.so.6                        0x00007f384229ad90
25 libc.so.6                        0x00007f384229ae40 __libc_start_main + 128
26 llc-opt                          0x0000563a3ced9b25
