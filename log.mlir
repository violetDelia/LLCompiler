[2024-09-26 08:01:59.214] [info] regist log module: mlir(lever:debug) -> 
[2024-09-26 08:01:59.214] [info] regist log module: utility(lever:debug) -> 
Args: /home/lfr/LLCompiler/.setuptools-cmake-build/bin/llc-opt --dump-pass-pipeline -o=/home/lfr/LLCompiler/out.mlir --log-lever=debug --log-root=C:codingLLCompilerlog --mlir-print-ir-tree-dir=/home/lfr/LLCompiler/it_tree --mlir-print-ir-after-all --operation-legalization --inline /home/lfr/LLCompiler/test/model_ir/resnet18.mlir -debug 
Load new dialect in Context builtin
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ShapedType)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::MemRefLayoutAttrInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::TypedAttr)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ElementsAttr)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::DistinctAttr)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::BytecodeOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::SymbolOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpAsmOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::RegionKindInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ConditionallySpeculatable)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::MemoryEffectOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ResourceBlobManagerDialectInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpAsmDialectInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::BytecodeDialectInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::AffineBinaryOpExprStorage)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::AffineConstantExprStorage)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::AffineDimExprStorage)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::AffineMapStorage)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::IntegerSetStorage)
Load new dialect in Context builtin
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ZeroOperands<mlir::TypeID::get<mlir::OpTrait::ZeroOperands>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::OneRegion<mlir::TypeID::get<mlir::OpTrait::OneRegion>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ZeroResults<mlir::TypeID::get<mlir::OpTrait::ZeroResults>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ZeroSuccessors<mlir::TypeID::get<mlir::OpTrait::ZeroSuccessors>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::NoRegionArguments<mlir::TypeID::get<mlir::OpTrait::NoRegionArguments>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::NoTerminator<mlir::TypeID::get<mlir::OpTrait::NoTerminator>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::SingleBlock<mlir::TypeID::get<mlir::OpTrait::SingleBlock>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::OpInvariants<mlir::TypeID::get<mlir::OpTrait::OpInvariants>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::BytecodeOpInterface::Trait<mlir::TypeID::get<mlir::BytecodeOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::AffineScope<mlir::TypeID::get<mlir::OpTrait::AffineScope>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::IsIsolatedFromAbove<mlir::TypeID::get<mlir::OpTrait::IsIsolatedFromAbove>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::SymbolTable<mlir::TypeID::get<mlir::OpTrait::SymbolTable>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::SymbolOpInterface::Trait<mlir::TypeID::get<mlir::SymbolOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpAsmOpInterface::Trait<mlir::TypeID::get<mlir::OpAsmOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::RegionKindInterface::Trait<mlir::TypeID::get<mlir::RegionKindInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::HasOnlyGraphRegion<mlir::TypeID::get<mlir::OpTrait::HasOnlyGraphRegion>()::Empty>)
Load new dialect in Context func
ImplicitTypeIDRegistry::lookupOrInsert(mlir::CallOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::SymbolUserOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::CallableOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::FunctionOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::RegionBranchTerminatorOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::DialectInlinerInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ConvertToLLVMPatternInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::bufferization::BufferizableOpInterface)
Load new dialect in Context cf
Load new dialect in Context arith
ImplicitTypeIDRegistry::lookupOrInsert(mlir::arith::ArithFastMathInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::VectorUnrollOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::InferTypeOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::InferIntRangeInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::arith::ArithIntegerOverflowFlagsInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::CastOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::arith::ArithRoundingModeInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::SelectLikeOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::bufferization::BufferDeallocationOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ValueBoundsOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::BranchOpInterface)
Load new dialect in Context llh
ImplicitTypeIDRegistry::lookupOrInsert(mlir::InferShapedTypeOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::LLHSymbolShapeOpInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ZeroRegions<mlir::TypeID::get<mlir::OpTrait::ZeroRegions>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::OneResult<mlir::TypeID::get<mlir::OpTrait::OneResult>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::OneTypedResult<mlir::Type>::Impl<mlir::TypeID::get<mlir::OpTrait::OneTypedResult<mlir::Type>::Impl>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::InferShapedTypeOpInterface::Trait<mlir::TypeID::get<mlir::InferShapedTypeOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::LLHSymbolShapeOpInterface::Trait<mlir::TypeID::get<mlir::LLHSymbolShapeOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::OneTypedResult<mlir::IntegerType>::Impl<mlir::TypeID::get<mlir::OpTrait::OneTypedResult<mlir::IntegerType>::Impl>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::InferTypeOpInterface::Trait<mlir::TypeID::get<mlir::InferTypeOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ConstantLike<mlir::TypeID::get<mlir::OpTrait::ConstantLike>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::AutomaticAllocationScope<mlir::TypeID::get<mlir::OpTrait::AutomaticAllocationScope>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::CallableOpInterface::Trait<mlir::TypeID::get<mlir::CallableOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::FunctionOpInterface::Trait<mlir::TypeID::get<mlir::FunctionOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::DataLayoutSpecInterface)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::IsTerminator<mlir::TypeID::get<mlir::OpTrait::IsTerminator>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::VariadicOperands<mlir::TypeID::get<mlir::OpTrait::VariadicOperands>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::HasParent<mlir::func::FuncOp>::Impl<mlir::TypeID::get<mlir::OpTrait::HasParent<mlir::func::FuncOp>::Impl>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::ConditionallySpeculatable::Trait<mlir::TypeID::get<mlir::ConditionallySpeculatable::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::AlwaysSpeculatableImplTrait<mlir::TypeID::get<mlir::OpTrait::AlwaysSpeculatableImplTrait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::MemoryEffectOpInterface::Trait<mlir::TypeID::get<mlir::MemoryEffectOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::MemRefsNormalizable<mlir::TypeID::get<mlir::OpTrait::MemRefsNormalizable>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::RegionBranchTerminatorOpInterface::Trait<mlir::TypeID::get<mlir::RegionBranchTerminatorOpInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ReturnLike<mlir::TypeID::get<mlir::OpTrait::ReturnLike>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::AtLeastNOperands<1>::Impl<mlir::TypeID::get<mlir::OpTrait::AtLeastNOperands<1>::Impl>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::NOperands<2>::Impl<mlir::TypeID::get<mlir::OpTrait::NOperands<2>::Impl>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::NOperands<5>::Impl<mlir::TypeID::get<mlir::OpTrait::NOperands<5>::Impl>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::OneOperand<mlir::TypeID::get<mlir::OpTrait::OneOperand>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::SameOperandsAndResultType<mlir::TypeID::get<mlir::OpTrait::SameOperandsAndResultType>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::SameOperandsAndResultRank<mlir::TypeID::get<mlir::OpTrait::SameOperandsAndResultRank>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::ResultsBroadcastableShape<mlir::TypeID::get<mlir::OpTrait::ResultsBroadcastableShape>()::Empty>)
Pass Manager with 2 passes:
builtin.module(ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::OpToOpPassAdaptor)
operation-legalization,inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 })

[2024-09-26 08:01:59.230] [info] ----- run in pass: Operationlegalization -----

//===-------------------------------------------===//
Legalizing operation : 'builtin.module'(0x56371d6753d0) {
  * Fold {
ImplicitTypeIDRegistry::lookupOrInsert(mlir::DialectFoldInterface)
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func.func'(0x56371d65b3b0) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d65c9d0) {
  %0 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d64e4e0) {
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d64e310) {
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d64ce50) {
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d62ffb0) {
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d663920) {
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d663ed0) {
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d663f90) {
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d664050) {
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d664110) {
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d65c0c0) {
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d65c230) {
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d664960) {
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d664f10) {
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d664fd0) {
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d665510) {
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d665a50) {
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d665b10) {
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6664d0) {
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d666590) {
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d667660) {
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d667720) {
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d667e20) {
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d667ee0) {
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d667fa0) {
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d668060) {
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d668120) {
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d664a20) {
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d668bb0) {
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d668c50) {
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d669200) {
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6692c0) {
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d669380) {
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6698c0) {
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66a9e0) {
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66aaa0) {
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ab60) {
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ac20) {
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ace0) {
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ada0) {
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ae60) {
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66af20) {
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66afe0) {
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b0a0) {
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b160) {
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b220) {
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66c3a0) {
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66c460) {
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66c520) {
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66d5f0) {
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66d6b0) {
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66d770) {
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66d830) {
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66d8f0) {
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66d9b0) {
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ddf0) {
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66deb0) {
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66df70) {
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66e030) {
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66e0f0) {
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66e1b0) {
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66e270) {
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66e330) {
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66e3f0) {
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f4c0) {
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f580) {
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f640) {
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f7b0) {
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f870) {
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f930) {
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66f9f0) {
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6702c0) {
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d670380) {
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d670440) {
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d670500) {
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6705c0) {
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d670680) {
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d670740) {
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d670800) {
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6718d0) {
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6721a0) {
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672260) {
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672320) {
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6723e0) {
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6724a0) {
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672560) {
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672620) {
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6726e0) {
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6727a0) {
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672860) {
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672920) {
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6729e0) {
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d672aa0) {
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d673b70) {
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b2e0) {
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b3a0) {
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b460) {
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b520) {
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b5e0) {
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b6a0) {
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b760) {
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b820) {
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b900) {
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66b9c0) {
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66ba80) {
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66bb40) {
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66bc00) {
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66bd60) {
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d66be20) {
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d676460) {
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6764e0) {
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d676ca0) {
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d676d60) {
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d676e20) {
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d676ee0) {
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d676fa0) {
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d677060) {
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d677120) {
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d6779f0) {
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d677ab0) {
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d677b70) {
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.weight'(0x56371d678c40) {
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.torch_symbolic_int'(0x56371d678d30) {
  %122 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.torch_symbolic_int'(0x56371d678df0) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d678ea0) {
  "llh.symbolic_bind"(%arg2, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d679100) {
  %124 = "llh.conv"(%arg2, %0) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d679200) {
  "llh.symbolic_bind"(%124, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d605e80) {
  %125 = "llh.batch_norm"(%124, %1, %2, %62, %63) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d679c00) {
  "llh.symbolic_bind"(%125, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d65f560) {
  %126 = "llh.relu"(%125) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d679d20) {
  "llh.symbolic_bind"(%126, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.max_pool'(0x56371d66f700) {
  %127 = "llh.max_pool"(%126) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67a760) {
  "llh.symbolic_bind"(%127, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67a850) {
  %128 = "llh.conv"(%127, %3) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67a950) {
  "llh.symbolic_bind"(%128, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67aa20) {
  %129 = "llh.batch_norm"(%128, %4, %5, %65, %66) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67ab70) {
  "llh.symbolic_bind"(%129, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d67ac40) {
  %130 = "llh.relu"(%129) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67ad20) {
  "llh.symbolic_bind"(%130, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67ae30) {
  %131 = "llh.conv"(%130, %6) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67af30) {
  "llh.symbolic_bind"(%131, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67b000) {
  %132 = "llh.batch_norm"(%131, %7, %8, %68, %69) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67b150) {
  "llh.symbolic_bind"(%132, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d67b220) {
  %133 = "llh.add"(%132, %127) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67b320) {
  "llh.symbolic_bind"(%133, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d67b3f0) {
  %134 = "llh.relu"(%133) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67b4d0) {
  "llh.symbolic_bind"(%134, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67bdf0) {
  %135 = "llh.conv"(%134, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67bef0) {
  "llh.symbolic_bind"(%135, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67bfc0) {
  %136 = "llh.batch_norm"(%135, %10, %11, %71, %72) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67c110) {
  "llh.symbolic_bind"(%136, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d67c1e0) {
  %137 = "llh.relu"(%136) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67c2c0) {
  "llh.symbolic_bind"(%137, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67c3d0) {
  %138 = "llh.conv"(%137, %12) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67c4d0) {
  "llh.symbolic_bind"(%138, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67c5a0) {
  %139 = "llh.batch_norm"(%138, %13, %14, %74, %75) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67c6f0) {
  "llh.symbolic_bind"(%139, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d67c7c0) {
  %140 = "llh.add"(%139, %134) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67c8c0) {
  "llh.symbolic_bind"(%140, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d67c990) {
  %141 = "llh.relu"(%140) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67ca70) {
  "llh.symbolic_bind"(%141, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67cb80) {
  %142 = "llh.conv"(%141, %15) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67cc80) {
  "llh.symbolic_bind"(%142, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67cd50) {
  %143 = "llh.batch_norm"(%142, %16, %17, %77, %78) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67cea0) {
  "llh.symbolic_bind"(%143, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d67cf70) {
  %144 = "llh.relu"(%143) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67d050) {
  "llh.symbolic_bind"(%144, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67d160) {
  %145 = "llh.conv"(%144, %18) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67d260) {
  "llh.symbolic_bind"(%145, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67d330) {
  %146 = "llh.batch_norm"(%145, %19, %20, %80, %81) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67d480) {
  "llh.symbolic_bind"(%146, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67ee20) {
  %147 = "llh.conv"(%141, %21) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67ef20) {
  "llh.symbolic_bind"(%147, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d67eff0) {
  %148 = "llh.batch_norm"(%147, %22, %23, %83, %84) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67f950) {
  "llh.symbolic_bind"(%148, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d67fa20) {
  %149 = "llh.add"(%146, %148) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67fb20) {
  "llh.symbolic_bind"(%149, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d67fbf0) {
  %150 = "llh.relu"(%149) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67fcd0) {
  "llh.symbolic_bind"(%150, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d67fde0) {
  %151 = "llh.conv"(%150, %24) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d67fee0) {
  "llh.symbolic_bind"(%151, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d6807c0) {
  %152 = "llh.batch_norm"(%151, %25, %26, %86, %87) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d681120) {
  "llh.symbolic_bind"(%152, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d6811f0) {
  %153 = "llh.relu"(%152) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6812d0) {
  "llh.symbolic_bind"(%153, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d6813e0) {
  %154 = "llh.conv"(%153, %27) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6814e0) {
  "llh.symbolic_bind"(%154, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d6815b0) {
  %155 = "llh.batch_norm"(%154, %28, %29, %89, %90) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d681700) {
  "llh.symbolic_bind"(%155, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d6817d0) {
  %156 = "llh.add"(%155, %150) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6818d0) {
  "llh.symbolic_bind"(%156, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d6819a0) {
  %157 = "llh.relu"(%156) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d681a80) {
  "llh.symbolic_bind"(%157, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d681b90) {
  %158 = "llh.conv"(%157, %30) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d681c90) {
  "llh.symbolic_bind"(%158, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d681d60) {
  %159 = "llh.batch_norm"(%158, %31, %32, %92, %93) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d681eb0) {
  "llh.symbolic_bind"(%159, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d682790) {
  %160 = "llh.relu"(%159) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d682870) {
  "llh.symbolic_bind"(%160, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d682980) {
  %161 = "llh.conv"(%160, %33) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d682a80) {
  "llh.symbolic_bind"(%161, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d682b50) {
  %162 = "llh.batch_norm"(%161, %34, %35, %95, %96) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d682ca0) {
  "llh.symbolic_bind"(%162, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d682db0) {
  %163 = "llh.conv"(%157, %36) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d682eb0) {
  "llh.symbolic_bind"(%163, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d682f80) {
  %164 = "llh.batch_norm"(%163, %37, %38, %98, %99) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6830d0) {
  "llh.symbolic_bind"(%164, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d6831a0) {
  %165 = "llh.add"(%162, %164) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d683ab0) {
  "llh.symbolic_bind"(%165, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d683b80) {
  %166 = "llh.relu"(%165) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d683c60) {
  "llh.symbolic_bind"(%166, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d683d70) {
  %167 = "llh.conv"(%166, %39) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d683e70) {
  "llh.symbolic_bind"(%167, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d683f40) {
  %168 = "llh.batch_norm"(%167, %40, %41, %101, %102) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d684090) {
  "llh.symbolic_bind"(%168, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d684160) {
  %169 = "llh.relu"(%168) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d684a50) {
  "llh.symbolic_bind"(%169, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d684b60) {
  %170 = "llh.conv"(%169, %42) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d684c60) {
  "llh.symbolic_bind"(%170, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d684d30) {
  %171 = "llh.batch_norm"(%170, %43, %44, %104, %105) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d684e80) {
  "llh.symbolic_bind"(%171, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d684f50) {
  %172 = "llh.add"(%171, %166) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d685050) {
  "llh.symbolic_bind"(%172, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d685120) {
  %173 = "llh.relu"(%172) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d685200) {
  "llh.symbolic_bind"(%173, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d685310) {
  %174 = "llh.conv"(%173, %45) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d685410) {
  "llh.symbolic_bind"(%174, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d6854e0) {
  %175 = "llh.batch_norm"(%174, %46, %47, %107, %108) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d685630) {
  "llh.symbolic_bind"(%175, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d685700) {
  %176 = "llh.relu"(%175) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6857e0) {
  "llh.symbolic_bind"(%176, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d6858f0) {
  %177 = "llh.conv"(%176, %48) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6859f0) {
  "llh.symbolic_bind"(%177, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d685ac0) {
  %178 = "llh.batch_norm"(%177, %49, %50, %110, %111) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d685c10) {
  "llh.symbolic_bind"(%178, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d685d20) {
  %179 = "llh.conv"(%173, %51) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d685e20) {
  "llh.symbolic_bind"(%179, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d685ef0) {
  %180 = "llh.batch_norm"(%179, %52, %53, %113, %114) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d686040) {
  "llh.symbolic_bind"(%180, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d686110) {
  %181 = "llh.add"(%178, %180) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d686210) {
  "llh.symbolic_bind"(%181, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d6862e0) {
  %182 = "llh.relu"(%181) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6863c0) {
  "llh.symbolic_bind"(%182, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d6864d0) {
  %183 = "llh.conv"(%182, %54) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6865d0) {
  "llh.symbolic_bind"(%183, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d6866a0) {
  %184 = "llh.batch_norm"(%183, %55, %56, %116, %117) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6867f0) {
  "llh.symbolic_bind"(%184, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d6868c0) {
  %185 = "llh.relu"(%184) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d6869a0) {
  "llh.symbolic_bind"(%185, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.conv'(0x56371d686ab0) {
  %186 = "llh.conv"(%185, %57) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d686bb0) {
  "llh.symbolic_bind"(%186, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.batch_norm'(0x56371d686c80) {
  %187 = "llh.batch_norm"(%186, %58, %59, %119, %120) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d686dd0) {
  "llh.symbolic_bind"(%187, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d686ea0) {
  %188 = "llh.add"(%187, %182) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d687fb0) {
  "llh.symbolic_bind"(%188, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.relu'(0x56371d688080) {
  %189 = "llh.relu"(%188) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d673c20) {
  "llh.symbolic_bind"(%189, %122, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.adaptive_average_pool'(0x56371d673cf0) {
  %190 = "llh.adaptive_average_pool"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d661ea0) {
  "llh.symbolic_bind"(%190, %122) <{expressions = affine_map<()[s0, s1] -> (s0, 512, 1, 1)>}> : (tensor<?x512x1x1xf32>, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.constant'(0x56371d673de0) {
  %191 = "llh.constant"() <{value = 1 : i64}> : () -> i64

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'llh.constant -> ()' {
Trying to match "{anonymous}::BraodcastableScalarToTensor"
"{anonymous}::BraodcastableScalarToTensor" result 0
  } -> FAILURE : pattern failed to match
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.flatten'(0x56371d673ea0) {
  %192 = "llh.flatten"(%190, %191) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d673fa0) {
  "llh.symbolic_bind"(%192, %122) <{expressions = affine_map<()[s0, s1] -> (s0, 512)>}> : (tensor<?x512xf32>, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.transpose'(0x56371d675110) {
  %193 = "llh.transpose"(%60) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.matmul'(0x56371d6751e0) {
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.add'(0x56371d6752d0) {
  %195 = "llh.add"(%194, %61) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llh.symbolic_bind'(0x56371d68b180) {
  "llh.symbolic_bind"(%195, %122) <{expressions = affine_map<()[s0, s1] -> (s0, 1000)>}> : (tensor<?x1000xf32>, i64) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func.return'(0x56371d65c890) {
  "func.return"(%195) : (tensor<?x1000xf32>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//
[2024-09-26 08:01:59.276] [info] ----- run out pass: Operationlegalization -----
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::PreservedAnalyses::AllAnalysesType)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::StorageUserTrait::IsMutable<mlir::TypeID::get<mlir::detail::StorageUserTrait::IsMutable>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::MemRefLayoutAttrInterface::Trait<mlir::TypeID::get<mlir::MemRefLayoutAttrInterface::Trait>()::Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::CallGraph)

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d65c9d0) {
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d64e4e0) {
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d64e310) {
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d64ce50) {
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d62ffb0) {
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d663920) {
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d663ed0) {
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d663f90) {
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664050) {
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664110) {
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d65c0c0) {
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d65c230) {
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664960) {
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664f10) {
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664fd0) {
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d665510) {
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d665a50) {
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d665b10) {
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6664d0) {
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d666590) {
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667660) {
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667720) {
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667e20) {
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667ee0) {
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667fa0) {
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668060) {
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668120) {
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664a20) {
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668bb0) {
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668c50) {
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d669200) {
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6692c0) {
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d669380) {
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6698c0) {
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66a9e0) {
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66aaa0) {
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ab60) {
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ac20) {
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ace0) {
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ada0) {
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ae60) {
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66af20) {
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66afe0) {
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b0a0) {
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b160) {
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b220) {
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66c3a0) {
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66c460) {
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66c520) {
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d5f0) {
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d6b0) {
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d770) {
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d830) {
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d8f0) {
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d9b0) {
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ddf0) {
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66deb0) {
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66df70) {
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e030) {
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e0f0) {
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e1b0) {
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e270) {
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e330) {
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e3f0) {
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f4c0) {
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::HasRecursiveMemoryEffects<mlir::TypeID::get<mlir::OpTrait::HasRecursiveMemoryEffects>()::Empty>)
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f580) {
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f640) {
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f7b0) {
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f870) {
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f930) {
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f9f0) {
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6702c0) {
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670380) {
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670440) {
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670500) {
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6705c0) {
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670680) {
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670740) {
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670800) {
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6718d0) {
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6721a0) {
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672260) {
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672320) {
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6723e0) {
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6724a0) {
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672560) {
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672620) {
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6726e0) {
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6727a0) {
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672860) {
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672920) {
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6729e0) {
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672aa0) {
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d673b70) {
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b2e0) {
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b3a0) {
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b460) {
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b520) {
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b5e0) {
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b6a0) {
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b760) {
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b820) {
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b900) {
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b9c0) {
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ba80) {
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66bb40) {
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66bc00) {
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66bd60) {
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66be20) {
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676460) {
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6764e0) {
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676ca0) {
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676d60) {
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676e20) {
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676ee0) {
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676fa0) {
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677060) {
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677120) {
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6779f0) {
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677ab0) {
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677b70) {
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d678c40) {
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678d30) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678df0) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d678ea0) {
  "llh.symbolic_bind"(%arg2, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d678ea0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d679100) {
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d679200) {
  "llh.symbolic_bind"(%125, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d679200)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d679100) {
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d605e80) {
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d679c00) {
  "llh.symbolic_bind"(%126, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d679c00)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d605e80) {
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d65f560) {
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d679d20) {
  "llh.symbolic_bind"(%127, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d679d20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d65f560) {
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.max_pool'(0x56371d66f700) {
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67a760) {
  "llh.symbolic_bind"(%128, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67a760)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67a850) {
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67a950) {
  "llh.symbolic_bind"(%129, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67a950)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67a850) {
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67aa20) {
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67ab70) {
  "llh.symbolic_bind"(%130, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67ab70)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67aa20) {
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67ac40) {
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67ad20) {
  "llh.symbolic_bind"(%131, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67ad20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67ac40) {
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67ae30) {
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67af30) {
  "llh.symbolic_bind"(%132, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67af30)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67ae30) {
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67b000) {
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67b150) {
  "llh.symbolic_bind"(%133, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67b150)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67b000) {
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67b220) {
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67b320) {
  "llh.symbolic_bind"(%134, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67b320)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67b220) {
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67b3f0) {
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67b4d0) {
  "llh.symbolic_bind"(%135, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67b4d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67bdf0) {
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67bef0) {
  "llh.symbolic_bind"(%136, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67bef0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67bdf0) {
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67bfc0) {
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67c110) {
  "llh.symbolic_bind"(%137, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67c110)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67bfc0) {
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67c1e0) {
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67c2c0) {
  "llh.symbolic_bind"(%138, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67c2c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67c1e0) {
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67c3d0) {
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67c4d0) {
  "llh.symbolic_bind"(%139, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67c4d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67c3d0) {
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67c5a0) {
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67c6f0) {
  "llh.symbolic_bind"(%140, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67c6f0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67c5a0) {
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67c7c0) {
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67c8c0) {
  "llh.symbolic_bind"(%141, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67c8c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67c7c0) {
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67c990) {
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67ca70) {
  "llh.symbolic_bind"(%142, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67ca70)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67cb80) {
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67cc80) {
  "llh.symbolic_bind"(%143, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67cc80)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67cb80) {
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67cd50) {
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67cea0) {
  "llh.symbolic_bind"(%144, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67cea0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67cd50) {
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67cf70) {
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67d050) {
  "llh.symbolic_bind"(%145, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67d050)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67cf70) {
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67d160) {
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67d260) {
  "llh.symbolic_bind"(%146, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67d260)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67d160) {
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67d330) {
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67d480) {
  "llh.symbolic_bind"(%147, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67d480)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67d330) {
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67ee20) {
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67ef20) {
  "llh.symbolic_bind"(%148, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67ef20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67ee20) {
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67eff0) {
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67f950) {
  "llh.symbolic_bind"(%149, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67f950)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67eff0) {
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67fa20) {
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67fb20) {
  "llh.symbolic_bind"(%150, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67fb20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67fa20) {
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67fbf0) {
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67fcd0) {
  "llh.symbolic_bind"(%151, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67fcd0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67fde0) {
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d67fee0) {
  "llh.symbolic_bind"(%152, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d67fee0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67fde0) {
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6807c0) {
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d681120) {
  "llh.symbolic_bind"(%153, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d681120)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6807c0) {
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6811f0) {
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6812d0) {
  "llh.symbolic_bind"(%154, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6812d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6811f0) {
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6813e0) {
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6814e0) {
  "llh.symbolic_bind"(%155, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6814e0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6813e0) {
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6815b0) {
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d681700) {
  "llh.symbolic_bind"(%156, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d681700)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6815b0) {
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6817d0) {
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6818d0) {
  "llh.symbolic_bind"(%157, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6818d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6817d0) {
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6819a0) {
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d681a80) {
  "llh.symbolic_bind"(%158, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d681a80)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d681b90) {
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d681c90) {
  "llh.symbolic_bind"(%159, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d681c90)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d681b90) {
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d681d60) {
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d681eb0) {
  "llh.symbolic_bind"(%160, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d681eb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d681d60) {
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d682790) {
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d682870) {
  "llh.symbolic_bind"(%161, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d682870)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d682790) {
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d682980) {
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d682a80) {
  "llh.symbolic_bind"(%162, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d682a80)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d682980) {
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d682b50) {
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d682ca0) {
  "llh.symbolic_bind"(%163, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d682ca0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d682b50) {
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d682db0) {
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d682eb0) {
  "llh.symbolic_bind"(%164, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d682eb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d682db0) {
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d682f80) {
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6830d0) {
  "llh.symbolic_bind"(%165, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6830d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d682f80) {
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6831a0) {
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d683ab0) {
  "llh.symbolic_bind"(%166, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d683ab0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6831a0) {
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d683b80) {
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d683c60) {
  "llh.symbolic_bind"(%167, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d683c60)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d683d70) {
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d683e70) {
  "llh.symbolic_bind"(%168, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d683e70)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d683d70) {
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d683f40) {
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d684090) {
  "llh.symbolic_bind"(%169, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d684090)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d683f40) {
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d684160) {
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d684a50) {
  "llh.symbolic_bind"(%170, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d684a50)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d684160) {
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d684b60) {
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d684c60) {
  "llh.symbolic_bind"(%171, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d684c60)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d684b60) {
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d684d30) {
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d684e80) {
  "llh.symbolic_bind"(%172, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d684e80)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d684d30) {
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d684f50) {
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d685050) {
  "llh.symbolic_bind"(%173, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d685050)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d684f50) {
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d685120) {
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d685200) {
  "llh.symbolic_bind"(%174, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d685200)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d685310) {
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d685410) {
  "llh.symbolic_bind"(%175, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d685410)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d685310) {
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6854e0) {
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d685630) {
  "llh.symbolic_bind"(%176, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d685630)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6854e0) {
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d685700) {
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6857e0) {
  "llh.symbolic_bind"(%177, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6857e0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d685700) {
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6858f0) {
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6859f0) {
  "llh.symbolic_bind"(%178, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6859f0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6858f0) {
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d685ac0) {
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d685c10) {
  "llh.symbolic_bind"(%179, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d685c10)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d685ac0) {
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d685d20) {
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d685e20) {
  "llh.symbolic_bind"(%180, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d685e20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d685d20) {
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d685ef0) {
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d686040) {
  "llh.symbolic_bind"(%181, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d686040)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d685ef0) {
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d686110) {
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d686210) {
  "llh.symbolic_bind"(%182, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d686210)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d686110) {
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6862e0) {
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6863c0) {
  "llh.symbolic_bind"(%183, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6863c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6864d0) {
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6865d0) {
  "llh.symbolic_bind"(%184, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6865d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6864d0) {
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6866a0) {
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6867f0) {
  "llh.symbolic_bind"(%185, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6867f0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6866a0) {
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6868c0) {
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d6869a0) {
  "llh.symbolic_bind"(%186, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d6869a0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6868c0) {
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d686ab0) {
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d686bb0) {
  "llh.symbolic_bind"(%187, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d686bb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d686ab0) {
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d686c80) {
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d686dd0) {
  "llh.symbolic_bind"(%188, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d686dd0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d686c80) {
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d686ea0) {
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d687fb0) {
  "llh.symbolic_bind"(%189, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d687fb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678df0) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d686ea0) {
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d688080) {
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d673c20) {
  "llh.symbolic_bind"(%190, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d673c20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678df0) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d688080) {
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.adaptive_average_pool'(0x56371d673cf0) {
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d661ea0) {
  "llh.symbolic_bind"(%191, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, 1, 1)>}> : (tensor<?x512x1x1xf32>, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d661ea0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.adaptive_average_pool'(0x56371d673cf0) {
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.constant'(0x56371d673de0) {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.flatten'(0x56371d673ea0) {
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d673fa0) {
  "llh.symbolic_bind"(%192, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512)>}> : (tensor<?x512xf32>, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d673fa0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678d30) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.flatten'(0x56371d673ea0) {
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.transpose'(0x56371d675110) {
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.matmul'(0x56371d6751e0) {
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6752d0) {
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56371d68b180) {
  "llh.symbolic_bind"(%195, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 1000)>}> : (tensor<?x1000xf32>, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56371d68b180)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678d30) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6752d0) {
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'func.return'(0x56371d65c890) {
  "func.return"(%195) : (tensor<?x1000xf32>) -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.constant'(0x56371d673de0) {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d65c9d0) {
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d64e4e0) {
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d64e310) {
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d64ce50) {
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d62ffb0) {
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d663920) {
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d663ed0) {
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d663f90) {
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664050) {
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664110) {
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d65c0c0) {
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d65c230) {
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664960) {
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664f10) {
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664fd0) {
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d665510) {
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d665a50) {
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d665b10) {
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6664d0) {
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d666590) {
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667660) {
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667720) {
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667e20) {
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667ee0) {
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d667fa0) {
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668060) {
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668120) {
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d664a20) {
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668bb0) {
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d668c50) {
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d669200) {
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6692c0) {
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d669380) {
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6698c0) {
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66a9e0) {
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66aaa0) {
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ab60) {
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ac20) {
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ace0) {
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ada0) {
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ae60) {
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66af20) {
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66afe0) {
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b0a0) {
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b160) {
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b220) {
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66c3a0) {
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66c460) {
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66c520) {
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d5f0) {
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d6b0) {
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d770) {
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d830) {
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d8f0) {
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66d9b0) {
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ddf0) {
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66deb0) {
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66df70) {
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e030) {
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e0f0) {
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e1b0) {
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e270) {
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e330) {
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66e3f0) {
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f4c0) {
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f580) {
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f640) {
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f7b0) {
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f870) {
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f930) {
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66f9f0) {
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6702c0) {
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670380) {
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670440) {
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670500) {
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6705c0) {
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670680) {
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670740) {
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d670800) {
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6718d0) {
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6721a0) {
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672260) {
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672320) {
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6723e0) {
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6724a0) {
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672560) {
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672620) {
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6726e0) {
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6727a0) {
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672860) {
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672920) {
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6729e0) {
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d672aa0) {
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d673b70) {
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b2e0) {
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b3a0) {
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b460) {
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b520) {
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b5e0) {
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b6a0) {
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b760) {
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b820) {
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b900) {
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66b9c0) {
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66ba80) {
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66bb40) {
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66bc00) {
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66bd60) {
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d66be20) {
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676460) {
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6764e0) {
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676ca0) {
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676d60) {
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676e20) {
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676ee0) {
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d676fa0) {
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677060) {
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677120) {
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d6779f0) {
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677ab0) {
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d677b70) {
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56371d678c40) {
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678d30) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x56371d678df0) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d679100) {
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d605e80) {
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d65f560) {
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.max_pool'(0x56371d66f700) {
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67a850) {
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67aa20) {
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67ac40) {
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67ae30) {
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67b000) {
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67b220) {
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67b3f0) {
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67bdf0) {
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67bfc0) {
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67c1e0) {
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67c3d0) {
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67c5a0) {
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67c7c0) {
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67c990) {
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67cb80) {
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67cd50) {
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67cf70) {
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67d160) {
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67d330) {
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67ee20) {
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d67eff0) {
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d67fa20) {
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d67fbf0) {
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d67fde0) {
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6807c0) {
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6811f0) {
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6813e0) {
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6815b0) {
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6817d0) {
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6819a0) {
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d681b90) {
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d681d60) {
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d682790) {
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d682980) {
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d682b50) {
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d682db0) {
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d682f80) {
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6831a0) {
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d683b80) {
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d683d70) {
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d683f40) {
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d684160) {
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d684b60) {
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d684d30) {
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d684f50) {
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d685120) {
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d685310) {
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6854e0) {
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d685700) {
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6858f0) {
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d685ac0) {
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d685d20) {
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d685ef0) {
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d686110) {
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6862e0) {
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d6864d0) {
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d6866a0) {
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d6868c0) {
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56371d686ab0) {
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56371d686c80) {
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d686ea0) {
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56371d688080) {
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.adaptive_average_pool'(0x56371d673cf0) {
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.flatten'(0x56371d673ea0) {
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.transpose'(0x56371d675110) {
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.matmul'(0x56371d6751e0) {
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56371d6752d0) {
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'func.return'(0x56371d65c890) {
  "func.return"(%195) : (tensor<?x1000xf32>) -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//
* Inliner: Initial calls in SCC are: {
}
* Inliner: Initial calls in SCC are: {
}
