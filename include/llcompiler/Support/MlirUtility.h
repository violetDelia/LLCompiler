//    Copyright 2024 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//

#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_MLIRUTILITY_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_MLIRUTILITY_H_

#define Loc_And_Context    \
  auto loc = op->getLoc(); \
  auto context = op->getContext();
// llh dialect
#define LLH_Constant(...) \
  rewriter.create<::mlir::llh::ConstantOp>(loc, __VA_ARGS__)
#define LLH_Add(...) rewriter.create<::mlir::llh::AddOp>(loc, __VA_ARGS__)
#define LLH_Mul(...) rewriter.create<::mlir::llh::MulOp>(loc, __VA_ARGS__)
#define LLH_Sub(...) rewriter.create<::mlir::llh::SubOp>(loc, __VA_ARGS__)
#define LLH_Div(...) rewriter.create<::mlir::llh::DivOp>(loc, __VA_ARGS__)
#define Sqrt(...) rewriter.create<::mlir::llh::SqrtOp>(loc, __VA_ARGS__)
#define Dim(...) rewriter.create<::mlir::llh::DimOp>(loc, __VA_ARGS__)
#define Conv(...) rewriter.create<::mlir::llh::ConvOp>(loc, __VA_ARGS__)
#define Reshape(...) rewriter.create<::mlir::llh::ReshapeOp>(loc, __VA_ARGS__)
#define BroadCastTo(...) \
  rewriter.create<::mlir::llh::BroadCastToOp>(loc, __VA_ARGS__)
#define StrideSlice(...) \
  rewriter.create<::mlir::llh::StrideSliceOp>(loc, __VA_ARGS__)
// vector dialect
#define Vec_Print(...) \
  rewriter.create<::mlir::vector::PrintOp>(loc, __VA_ARGS__)
// index dialect
#define IndexCast(...) \
  rewriter.create<::mlir::arith::IndexCastOp>(loc, __VA_ARGS__)
// memref dialect
#define Mem_Cast(...) rewriter.create<::mlir::memref::CastOp>(loc, __VA_ARGS__)
#define Mem_CollapseShape(...) \
  rewriter.create<::mlir::memref::CollapseShapeOp>(loc, __VA_ARGS__)
#define Mem_Alloc(...) \
  rewriter.create<::mlir::memref::AllocOp>(loc, __VA_ARGS__)
// func dielct
#define Fun_Call(...) rewriter.create<::mlir::func::CallOp>(loc, __VA_ARGS__)
// tensor dialect
#define FromElements(...) \
  rewriter.create<::mlir::tensor::FromElementsOp>(loc, __VA_ARGS__)
#define Tensor_Dim(...) rewriter.create<::mlir::tensor::DimOp>(loc, __VA_ARGS__)
#define Tensor_Reshape(...) \
  rewriter.create<::mlir::tensor::ReshapeOp>(loc, __VA_ARGS__)
#define Tensor_Empty(...) \
  rewriter.create<::mlir::tensor::EmptyOp>(loc, __VA_ARGS__)
#define Tensor_Extract(...) \
  rewriter.create<::mlir::tensor::ExtractOp>(loc, __VA_ARGS__)
// arith dialect
#define ConstantIndex(...) \
  rewriter.create<::mlir::arith::ConstantIndexOp>(loc, __VA_ARGS__)
#define Arith_SubI(...) rewriter.create<::mlir::arith::SubIOp>(loc, __VA_ARGS__)
#define Arith_CeilDivUI(...) \
  rewriter.create<::mlir::arith::CeilDivUIOp>(loc, __VA_ARGS__)
// bufferization dialect
#define ToMemref(...) \
  rewriter.create<::mlir::bufferization::ToMemrefOp>(loc, __VA_ARGS__)
// shape dialect
#define Shape_Dim(...) rewriter.create<::mlir::shape::DimOp>(loc, __VA_ARGS__)
// stablehlo dialect
#define HLO_Constant(...) \
  rewriter.create<::mlir::stablehlo::ConstantOp>(loc, __VA_ARGS__)
#define HLO_Reduce(...) \
  rewriter.create<::mlir::stablehlo::ReduceOp>(loc, __VA_ARGS__)
#define HLO_Max(...) rewriter.create<::mlir::stablehlo::MaxOp>(loc, __VA_ARGS__)
#define HLO_Min(...) rewriter.create<::mlir::stablehlo::MinOp>(loc, __VA_ARGS__)
#define HLO_Add(...) rewriter.create<::mlir::stablehlo::AddOp>(loc, __VA_ARGS__)
#define HLO_Return(...) \
  rewriter.create<::mlir::stablehlo::ReturnOp>(loc, __VA_ARGS__)
#define HLO_DynamicBroadcastInDim(...) \
  rewriter.create<::mlir::stablehlo::DynamicBroadcastInDimOp>(loc, __VA_ARGS__)
#define HLO_ReduceWindow(...) \
  rewriter.create<::mlir::stablehlo::ReduceWindowOp>(loc, __VA_ARGS__)
#define HLO_RealDynamicSlice(...) \
  rewriter.create<::mlir::stablehlo::RealDynamicSliceOp>(loc, __VA_ARGS__)

// type
#define I64_Ty ::mlir::IntegerType::get(context, 64)
#define I32_Ty ::mlir::IntegerType::get(context, 32)
#define I16_Ty ::mlir::IntegerType::get(context, 16)
#define I8_Ty ::mlir::IntegerType::get(context, 8)
#define Ii_Ty ::mlir::IntegerType::get(context, 1)
#define UI64_Ty                         \
  ::mlir::IntegerType::get(context, 64, \
                           mlir::IntegerType::SignednessSemantics::Unsigned)
#define UI32_Ty                         \
  ::mlir::IntegerType::get(context, 32, \
                           mlir::IntegerType::SignednessSemantics::Unsigned)
#define UI16_Ty                         \
  ::mlir::IntegerType::get(context, 16, \
                           mlir::IntegerType::SignednessSemantics::Unsigned)
#define UI8_Ty                         \
  ::mlir::IntegerType::get(context, 8, \
                           mlir::IntegerType::SignednessSemantics::Unsigned)
#define Index_Ty ::mlir::IndexType::get(context)
#define Function_Ty(...) ::mlir::FunctionType::get(context, __VA_ARGS__)
// attrs
#define I64_Attr(value) \
  ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, 64), value)
#define I32_Attr(value) \
  ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, 32), value)
#define I16_Attr(value) \
  ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, 16), value)
#define I8_Attr(value) \
  ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, 8), value)
#define Ii_Attr(value) \
  ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, 1), value)
#define Index_Attr(value) \
  ::mlir::IntegerAttr::get(::mlir::IndexType::get(context), value)
#define UI64_Attr(value)                                                  \
  ::mlir::IntegerAttr::get(                                               \
      ::mlir::IntegerType::get(                                           \
          context, 64, mlir::IntegerType::SignednessSemantics::Unsigned), \
      value)
#define UI32_Attr(value)                                                  \
  ::mlir::IntegerAttr::get(                                               \
      ::mlir::IntegerType::get(                                           \
          context, 32, mlir::IntegerType::SignednessSemantics::Unsigned), \
      value)
#define UI16_Attr(value)                                                  \
  ::mlir::IntegerAttr::get(                                               \
      ::mlir::IntegerType::get(                                           \
          context, 16, mlir::IntegerType::SignednessSemantics::Unsigned), \
      value)
#define UI8_Attr(value)                                                  \
  ::mlir::IntegerAttr::get(                                              \
      ::mlir::IntegerType::get(                                          \
          context, 8, mlir::IntegerType::SignednessSemantics::Unsigned), \
      value)
#define None_Overflow_Attr              \
  arith::IntegerOverflowFlagsAttr::get( \
      context, ::mlir::arith::IntegerOverflowFlags::none)
#define None_FastMath_Attr \
  arith::FastMathFlagsAttr::get(context, ::mlir::arith::FastMathFlags::none)

#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_MLIRUTILITY_H_