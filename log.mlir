[2024-09-27 00:52:14.680] [info] regist log module: mlir(lever:debug) -> 
[2024-09-27 00:52:14.680] [info] regist log module: utility(lever:debug) -> 
Args: /home/lfr/LLCompiler/.setuptools-cmake-build/bin/llc-opt --dump-pass-pipeline -o=/home/lfr/LLCompiler/out.mlir --log-lever=debug --log-root=C:codingLLCompilerlog --mlir-print-ir-tree-dir=/home/lfr/LLCompiler/it_tree --mlir-print-ir-after-all --operation-legalization --inline /home/lfr/LLCompiler/test/model_ir/resnet18.mlir --debug 
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

[2024-09-27 00:52:14.682] [info] ----- run in pass: Operationlegalization -----
ImplicitTypeIDRegistry::lookupOrInsert(mlir::DialectFoldInterface)

//===-------------------------------------------===//
Processing operation : 'func.return'(0x562712014f90) {
  "func.return"(%195) : (tensor<?x1000xf32>) -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271202d680) {
  "llh.symbolic_bind"(%195, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 1000)>}> : (tensor<?x1000xf32>, i64) -> ()

ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::HasRecursiveMemoryEffects<mlir::TypeID::get<mlir::OpTrait::HasRecursiveMemoryEffects>()::Empty>)
  ** Erase   : 'llh.symbolic_bind'(0x56271202d680)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271202d580) {
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.matmul'(0x56271202d490) {
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.transpose'(0x56271202d3c0) {
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271202c2c0) {
  "llh.symbolic_bind"(%192, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512)>}> : (tensor<?x512xf32>, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271202c2c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.flatten'(0x56271202c1c0) {
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.constant'(0x56271202c100) {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64


  * Pattern {anonymous}::BraodcastableScalarToTensor : 'llh.constant -> ()' {
Trying to match "{anonymous}::BraodcastableScalarToTensor"
"{anonymous}::BraodcastableScalarToTensor" result 0
  } -> failure : pattern failed to match
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271201a0e0) {
  "llh.symbolic_bind"(%191, %123) <{expressions = affine_map<()[s0, s1] -> (s0, 512, 1, 1)>}> : (tensor<?x512x1x1xf32>, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271201a0e0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.adaptive_average_pool'(0x56271202c010) {
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271202bf40) {
  "llh.symbolic_bind"(%190, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271202bf40)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712041030) {
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712040f60) {
  "llh.symbolic_bind"(%189, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712040f60)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203fe50) {
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203fd80) {
  "llh.symbolic_bind"(%188, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203fd80)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203fc30) {
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203fb60) {
  "llh.symbolic_bind"(%187, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203fb60)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203fa60) {
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203f950) {
  "llh.symbolic_bind"(%186, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203f950)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203f870) {
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203f7a0) {
  "llh.symbolic_bind"(%185, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203f7a0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203f650) {
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203f580) {
  "llh.symbolic_bind"(%184, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203f580)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203f480) {
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203f370) {
  "llh.symbolic_bind"(%183, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203f370)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203f290) {
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203f1c0) {
  "llh.symbolic_bind"(%182, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203f1c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203f0c0) {
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203eff0) {
  "llh.symbolic_bind"(%181, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203eff0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203eea0) {
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203edd0) {
  "llh.symbolic_bind"(%180, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203edd0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203ecd0) {
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203ebc0) {
  "llh.symbolic_bind"(%179, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203ebc0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203ea70) {
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203e190) {
  "llh.symbolic_bind"(%178, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203e190)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203e090) {
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203df80) {
  "llh.symbolic_bind"(%177, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203df80)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203dea0) {
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203ddd0) {
  "llh.symbolic_bind"(%176, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203ddd0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203dc80) {
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203dbb0) {
  "llh.symbolic_bind"(%175, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203dbb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203dab0) {
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203d9a0) {
  "llh.symbolic_bind"(%174, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203d9a0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203d8c0) {
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203d7f0) {
  "llh.symbolic_bind"(%173, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203d7f0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203d6f0) {
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203d620) {
  "llh.symbolic_bind"(%172, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203d620)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203d4d0) {
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203d400) {
  "llh.symbolic_bind"(%171, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203d400)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203d300) {
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203d1f0) {
  "llh.symbolic_bind"(%170, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203d1f0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203d110) {
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203d040) {
  "llh.symbolic_bind"(%169, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203d040)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203cef0) {
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203ce20) {
  "llh.symbolic_bind"(%168, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203ce20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203cd20) {
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203cc10) {
  "llh.symbolic_bind"(%167, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203cc10)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203cb30) {
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203c250) {
  "llh.symbolic_bind"(%166, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203c250)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203c150) {
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203c080) {
  "llh.symbolic_bind"(%165, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203c080)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203bf30) {
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203be60) {
  "llh.symbolic_bind"(%164, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203be60)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203bd60) {
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203bc50) {
  "llh.symbolic_bind"(%163, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203bc50)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203bb00) {
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203ba30) {
  "llh.symbolic_bind"(%162, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203ba30)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203b930) {
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203b820) {
  "llh.symbolic_bind"(%161, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203b820)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203b740) {
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203b670) {
  "llh.symbolic_bind"(%160, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203b670)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203b520) {
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203b450) {
  "llh.symbolic_bind"(%159, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203b450)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203b350) {
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203aa30) {
  "llh.symbolic_bind"(%158, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203aa30)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203a950) {
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203a880) {
  "llh.symbolic_bind"(%157, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203a880)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203a780) {
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203a6b0) {
  "llh.symbolic_bind"(%156, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203a6b0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203a560) {
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203a490) {
  "llh.symbolic_bind"(%155, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203a490)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203a390) {
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x56271203a280) {
  "llh.symbolic_bind"(%154, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x56271203a280)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203a1a0) {
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120398c0) {
  "llh.symbolic_bind"(%153, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120398c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712039770) {
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712038e90) {
  "llh.symbolic_bind"(%152, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712038e90)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712038580) {
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712038470) {
  "llh.symbolic_bind"(%151, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712038470)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712038390) {
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120382c0) {
  "llh.symbolic_bind"(%150, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120382c0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x5627120381c0) {
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120380f0) {
  "llh.symbolic_bind"(%149, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120380f0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712037fa0) {
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712037ed0) {
  "llh.symbolic_bind"(%148, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712037ed0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712037dd0) {
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036c40) {
  "llh.symbolic_bind"(%147, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036c40)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712036af0) {
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036a20) {
  "llh.symbolic_bind"(%146, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036a20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712036920) {
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036810) {
  "llh.symbolic_bind"(%145, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036810)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712036730) {
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036660) {
  "llh.symbolic_bind"(%144, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036660)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712036510) {
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036440) {
  "llh.symbolic_bind"(%143, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036440)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712036340) {
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036230) {
  "llh.symbolic_bind"(%142, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036230)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712036150) {
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712036080) {
  "llh.symbolic_bind"(%141, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712036080)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x562712035f80) {
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712035eb0) {
  "llh.symbolic_bind"(%140, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712035eb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712035d60) {
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712034c70) {
  "llh.symbolic_bind"(%139, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712034c70)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712034b70) {
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712034a60) {
  "llh.symbolic_bind"(%138, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712034a60)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712034980) {
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120348b0) {
  "llh.symbolic_bind"(%137, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120348b0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712034760) {
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712034690) {
  "llh.symbolic_bind"(%136, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712034690)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712034590) {
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712034480) {
  "llh.symbolic_bind"(%135, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712034480)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x5627120343a0) {
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120342d0) {
  "llh.symbolic_bind"(%134, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120342d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x5627120341d0) {
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712034100) {
  "llh.symbolic_bind"(%133, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712034100)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712033fb0) {
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712033ee0) {
  "llh.symbolic_bind"(%132, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712033ee0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712033de0) {
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712033cd0) {
  "llh.symbolic_bind"(%131, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712033cd0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712033bf0) {
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712033b20) {
  "llh.symbolic_bind"(%130, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712033b20)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x5627120339d0) {
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712033900) {
  "llh.symbolic_bind"(%129, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712033900)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712033800) {
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712033710) {
  "llh.symbolic_bind"(%128, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712033710)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.max_pool'(0x562712028a40) {
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712032cd0) {
  "llh.symbolic_bind"(%127, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712032cd0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712017c20) {
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x562712032bb0) {
  "llh.symbolic_bind"(%126, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x562712032bb0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562711fbee80) {
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120321b0) {
  "llh.symbolic_bind"(%125, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120321b0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x5627120320b0) {
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_bind'(0x5627120319d0) {
  "llh.symbolic_bind"(%arg2, %123, %124) <{expressions = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()

  ** Erase   : 'llh.symbolic_bind'(0x5627120319d0)
} -> success : operation is trivially dead
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x562712031920) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64


  * Pattern {anonymous}::replaceTorchSymbolicIntOp : 'llh.torch_symbolic_int -> ()' {
Trying to match "{anonymous}::replaceTorchSymbolicIntOp"
[2024-09-27 00:52:14.701] [info] symbol: s0
ImplicitTypeIDRegistry::lookupOrInsert(mlir::llh::detail::SymbolicIntOpGenericAdaptorBase::Properties)
"{anonymous}::replaceTorchSymbolicIntOp" result 1
  } -> success : pattern applied successfully
// *** IR Dump After Pattern Application ***
func.func @main(%arg0: i64, %arg1: i64, %arg2: tensor<?x3x?x?xf32>) -> tensor<?x1000xf32> {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> {symbol_generated} : () -> i64
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
  return %195 : tensor<?x1000xf32>
}


} -> success : pattern matched
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x562712031860) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64


  * Pattern {anonymous}::replaceTorchSymbolicIntOp : 'llh.torch_symbolic_int -> ()' {
Trying to match "{anonymous}::replaceTorchSymbolicIntOp"
[2024-09-27 00:52:14.729] [info] symbol: s1
"{anonymous}::replaceTorchSymbolicIntOp" result 1
  } -> success : pattern applied successfully
// *** IR Dump After Pattern Application ***
func.func @main(%arg0: i64, %arg1: i64, %arg2: tensor<?x3x?x?xf32>) -> tensor<?x1000xf32> {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> {symbol_generated} : () -> i64
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> {symbol_generated} : () -> i64
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
  return %195 : tensor<?x1000xf32>
}


} -> success : pattern matched
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712031770) {
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120306a0) {
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120305e0) {
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030520) {
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030460) {
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120303a0) {
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120302e0) {
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030220) {
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030160) {
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120300a0) {
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ffe0) {
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202f820) {
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202f7a0) {
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024d60) {
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024ca0) {
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024b40) {
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024a80) {
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120249c0) {
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024900) {
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024840) {
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024760) {
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120246a0) {
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120245e0) {
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024520) {
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024460) {
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120243a0) {
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120242e0) {
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024220) {
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202be90) {
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202adc0) {
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ad00) {
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ac40) {
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ab80) {
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202aac0) {
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202aa00) {
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a940) {
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a880) {
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a7c0) {
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a700) {
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a640) {
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a580) {
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a4c0) {
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a400) {
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029330) {
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029270) {
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120291b0) {
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120290f0) {
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029030) {
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028f70) {
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028eb0) {
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028df0) {
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028d30) {
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028c70) {
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028bb0) {
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028af0) {
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028980) {
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120288c0) {
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028800) {
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027730) {
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027670) {
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120275b0) {
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120274f0) {
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027430) {
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027370) {
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120272b0) {
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120271f0) {
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027130) {
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120264e0) {
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026420) {
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026360) {
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120262a0) {
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026170) {
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120260b0) {
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024fe0) {
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024f20) {
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024e60) {
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024160) {
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023890) {
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120237d0) {
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023710) {
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023650) {
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023590) {
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120234d0) {
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023410) {
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023350) {
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023290) {
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712022d50) {
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712022c90) {
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021b70) {
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021ab0) {
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120219f0) {
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021930) {
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120213f0) {
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021350) {
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d0d0) {
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120210d0) {
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021010) {
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020f50) {
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020e90) {
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020dd0) {
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120206d0) {
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201fd10) {
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201ec40) {
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201eb80) {
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e640) {
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e580) {
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e040) {
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d680) {
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d1b0) {
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d030) {
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712014d40) {
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712014c40) {
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201cc50) {
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c710) {
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c650) {
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c110) {
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201bfe0) {
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562711fe8fb0) {
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120059d0) {
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712006e90) {
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712007060) {
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'func.func'(0x562712014d90) {
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120150d0) {
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'func.return'(0x562712014f90) {
  "func.return"(%195) : (tensor<?x1000xf32>) -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271202d580) {
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.matmul'(0x56271202d490) {
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.transpose'(0x56271202d3c0) {
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.flatten'(0x56271202c1c0) {
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.adaptive_average_pool'(0x56271202c010) {
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712041030) {
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203fe50) {
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203fc30) {
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203fa60) {
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203f870) {
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203f650) {
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203f480) {
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203f290) {
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203f0c0) {
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203eea0) {
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203ecd0) {
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203ea70) {
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203e090) {
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203dea0) {
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203dc80) {
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203dab0) {
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203d8c0) {
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203d6f0) {
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203d4d0) {
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203d300) {
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203d110) {
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203cef0) {
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203cd20) {
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203cb30) {
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203c150) {
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203bf30) {
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203bd60) {
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203bb00) {
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203b930) {
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203b740) {
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203b520) {
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203b350) {
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203a950) {
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203a780) {
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203a560) {
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203a390) {
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203a1a0) {
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712039770) {
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712038580) {
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712038390) {
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x5627120381c0) {
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712037fa0) {
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712037dd0) {
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712036af0) {
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712036920) {
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712036730) {
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712036510) {
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712036340) {
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712036150) {
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x562712035f80) {
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712035d60) {
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712034b70) {
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712034980) {
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712034760) {
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712034590) {
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x5627120343a0) {
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x5627120341d0) {
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712033fb0) {
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712033de0) {
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712033bf0) {
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x5627120339d0) {
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712033800) {
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.max_pool'(0x562712028a40) {
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712017c20) {
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562711fbee80) {
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x5627120320b0) {
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x562712031920) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> {symbol_generated} : () -> i64


  * Pattern {anonymous}::replaceTorchSymbolicIntOp : 'llh.torch_symbolic_int -> ()' {
Trying to match "{anonymous}::replaceTorchSymbolicIntOp"
"{anonymous}::replaceTorchSymbolicIntOp" result 0
  } -> failure : pattern failed to match
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x562712031860) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> {symbol_generated} : () -> i64


  * Pattern {anonymous}::replaceTorchSymbolicIntOp : 'llh.torch_symbolic_int -> ()' {
Trying to match "{anonymous}::replaceTorchSymbolicIntOp"
"{anonymous}::replaceTorchSymbolicIntOp" result 0
  } -> failure : pattern failed to match
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712031770) {
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120306a0) {
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120305e0) {
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030520) {
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030460) {
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120303a0) {
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120302e0) {
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030220) {
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030160) {
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120300a0) {
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ffe0) {
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202f820) {
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202f7a0) {
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024d60) {
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024ca0) {
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024b40) {
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024a80) {
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120249c0) {
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024900) {
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024840) {
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024760) {
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120246a0) {
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120245e0) {
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024520) {
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024460) {
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120243a0) {
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120242e0) {
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024220) {
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202be90) {
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202adc0) {
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ad00) {
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ac40) {
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ab80) {
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202aac0) {
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202aa00) {
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a940) {
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a880) {
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a7c0) {
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a700) {
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a640) {
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a580) {
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a4c0) {
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a400) {
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029330) {
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029270) {
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120291b0) {
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120290f0) {
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029030) {
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028f70) {
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028eb0) {
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028df0) {
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028d30) {
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028c70) {
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028bb0) {
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028af0) {
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028980) {
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120288c0) {
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028800) {
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027730) {
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027670) {
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120275b0) {
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120274f0) {
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027430) {
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027370) {
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120272b0) {
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120271f0) {
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027130) {
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120264e0) {
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026420) {
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026360) {
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120262a0) {
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026170) {
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120260b0) {
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024fe0) {
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024f20) {
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024e60) {
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024160) {
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023890) {
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120237d0) {
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023710) {
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023650) {
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023590) {
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120234d0) {
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023410) {
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023350) {
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023290) {
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712022d50) {
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712022c90) {
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021b70) {
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021ab0) {
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120219f0) {
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021930) {
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120213f0) {
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021350) {
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d0d0) {
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120210d0) {
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021010) {
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020f50) {
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020e90) {
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020dd0) {
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120206d0) {
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201fd10) {
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201ec40) {
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201eb80) {
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e640) {
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e580) {
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e040) {
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d680) {
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d1b0) {
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d030) {
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712014d40) {
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712014c40) {
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201cc50) {
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c710) {
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c650) {
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c110) {
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201bfe0) {
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562711fe8fb0) {
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120059d0) {
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712006e90) {
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712007060) {
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120150d0) {
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'func.func'(0x562712014d90) {
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.constant'(0x56271202c100) {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64


  * Pattern {anonymous}::BraodcastableScalarToTensor : 'llh.constant -> ()' {
Trying to match "{anonymous}::BraodcastableScalarToTensor"
"{anonymous}::BraodcastableScalarToTensor" result 0
  } -> failure : pattern failed to match
} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_int'(0x56271201af70) {
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.symbolic_int'(0x562712022c30) {
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//
[2024-09-27 00:52:14.820] [info] ----- run out pass: Operationlegalization -----
ImplicitTypeIDRegistry::lookupOrInsert(mlir::detail::PreservedAnalyses::AllAnalysesType)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::CallGraph)

//===-------------------------------------------===//
Processing operation : 'llh.constant'(0x56271202c100) {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120150d0) {
  %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712007060) {
  %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712006e90) {
  %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120059d0) {
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562711fe8fb0) {
  %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201bfe0) {
  %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c110) {
  %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c650) {
  %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201c710) {
  %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201cc50) {
  %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712014c40) {
  %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712014d40) {
  %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d030) {
  %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d1b0) {
  %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d680) {
  %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e040) {
  %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e580) {
  %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201e640) {
  %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201eb80) {
  %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201ec40) {
  %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201fd10) {
  %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120206d0) {
  %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020dd0) {
  %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020e90) {
  %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712020f50) {
  %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021010) {
  %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120210d0) {
  %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271201d0d0) {
  %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021350) {
  %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120213f0) {
  %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021930) {
  %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120219f0) {
  %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021ab0) {
  %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712021b70) {
  %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712022c90) {
  %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712022d50) {
  %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023290) {
  %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023350) {
  %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023410) {
  %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120234d0) {
  %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023590) {
  %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023650) {
  %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023710) {
  %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120237d0) {
  %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712023890) {
  %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024160) {
  %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024e60) {
  %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024f20) {
  %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024fe0) {
  %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120260b0) {
  %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026170) {
  %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120262a0) {
  %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026360) {
  %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712026420) {
  %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120264e0) {
  %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027130) {
  %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120271f0) {
  %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120272b0) {
  %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027370) {
  %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027430) {
  %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120274f0) {
  %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120275b0) {
  %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027670) {
  %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712027730) {
  %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028800) {
  %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120288c0) {
  %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028980) {
  %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028af0) {
  %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028bb0) {
  %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028c70) {
  %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028d30) {
  %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028df0) {
  %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028eb0) {
  %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712028f70) {
  %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029030) {
  %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120290f0) {
  %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120291b0) {
  %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029270) {
  %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712029330) {
  %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a400) {
  %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a4c0) {
  %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a580) {
  %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a640) {
  %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a700) {
  %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a7c0) {
  %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a880) {
  %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202a940) {
  %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202aa00) {
  %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202aac0) {
  %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ab80) {
  %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ac40) {
  %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ad00) {
  %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202adc0) {
  %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202be90) {
  %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024220) {
  %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120242e0) {
  %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120243a0) {
  %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024460) {
  %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024520) {
  %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120245e0) {
  %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120246a0) {
  %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024760) {
  %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024840) {
  %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024900) {
  %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120249c0) {
  %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024a80) {
  %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024b40) {
  %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024ca0) {
  %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712024d60) {
  %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202f7a0) {
  %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202f820) {
  %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x56271202ffe0) {
  %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120300a0) {
  %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030160) {
  %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030220) {
  %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120302e0) {
  %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120303a0) {
  %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030460) {
  %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712030520) {
  %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120305e0) {
  %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x5627120306a0) {
  %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.weight'(0x562712031770) {
  %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x562712031860) {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> {symbol_generated} : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.torch_symbolic_int'(0x562712031920) {
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> {symbol_generated} : () -> i64

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x5627120320b0) {
  %125 = "llh.conv"(%arg2, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562711fbee80) {
  %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712017c20) {
  %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.max_pool'(0x562712028a40) {
  %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712033800) {
  %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x5627120339d0) {
  %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712033bf0) {
  %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712033de0) {
  %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712033fb0) {
  %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x5627120341d0) {
  %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x5627120343a0) {
  %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712034590) {
  %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712034760) {
  %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712034980) {
  %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712034b70) {
  %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712035d60) {
  %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x562712035f80) {
  %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712036150) {
  %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712036340) {
  %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712036510) {
  %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712036730) {
  %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712036920) {
  %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712036af0) {
  %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712037dd0) {
  %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712037fa0) {
  %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x5627120381c0) {
  %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712038390) {
  %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x562712038580) {
  %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x562712039770) {
  %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203a1a0) {
  %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203a390) {
  %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203a560) {
  %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203a780) {
  %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203a950) {
  %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203b350) {
  %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203b520) {
  %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203b740) {
  %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203b930) {
  %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203bb00) {
  %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203bd60) {
  %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203bf30) {
  %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203c150) {
  %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203cb30) {
  %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203cd20) {
  %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203cef0) {
  %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203d110) {
  %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203d300) {
  %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203d4d0) {
  %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203d6f0) {
  %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203d8c0) {
  %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203dab0) {
  %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203dc80) {
  %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203dea0) {
  %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203e090) {
  %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203ea70) {
  %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203ecd0) {
  %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203eea0) {
  %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203f0c0) {
  %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203f290) {
  %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203f480) {
  %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203f650) {
  %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x56271203f870) {
  %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.conv'(0x56271203fa60) {
  %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.batch_norm'(0x56271203fc30) {
  %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271203fe50) {
  %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.relu'(0x562712041030) {
  %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.adaptive_average_pool'(0x56271202c010) {
  %191 = "llh.adaptive_average_pool"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.flatten'(0x56271202c1c0) {
  %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.transpose'(0x56271202d3c0) {
  %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.matmul'(0x56271202d490) {
  %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'llh.add'(0x56271202d580) {
  %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>

} -> failure : pattern failed to match
//===-------------------------------------------===//

//===-------------------------------------------===//
Processing operation : 'func.return'(0x562712014f90) {
  "func.return"(%195) : (tensor<?x1000xf32>) -> ()

} -> failure : pattern failed to match
//===-------------------------------------------===//
* Inliner: Initial calls in SCC are: {
}
* Inliner: Initial calls in SCC are: {
}
