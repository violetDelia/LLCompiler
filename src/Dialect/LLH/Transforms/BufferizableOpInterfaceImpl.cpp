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
#include "llcompiler/Dialect/LLH/Transforms/BufferizableOpInterfaceImpl.h"

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/MlirUtility.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace mlir {
namespace llh {
namespace {

struct PrintOpInterface
    : public BufferizableOpInterface::ExternalModel<PrintOpInterface,
                                                    llh::PrintOp> {
 public:
  bool bufferizesToAllocation(Operation *op, ::mlir::Value value) const {
    return false;
  };
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }
  bool bufferizesToElementwiseAccess(
      Operation *op, const ::mlir::bufferization::AnalysisState &state,
      ArrayRef<OpOperand *> opOperands) const {
    return true;
  };
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  };
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    Loc_And_Context;;
    auto print = cast<llh::PrintOp>(op);
    auto input = print.getInput();
    auto input_type = input.getType();
    if (auto tensor_type =
            llvm::dyn_cast_or_null<mlir::TensorType>(input_type)) {
      auto memref_type = bufferization::getBufferType(input, options);
      auto to_memref = ToMemref(*memref_type, input, true);
      replaceOpWithNewBufferizedOp<llh::PrintOp>(
          rewriter, op, to_memref->getResult(0), print.getPrefixDescription());
      return llvm::success();
    }
    return llvm::failure();
  }
};

}  // namespace
}  // namespace llh
}  // namespace mlir
void mlir::llh::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, llh::LLHDialect *dialect) {
    PrintOp::attachInterface<PrintOpInterface>(*ctx);
  });
}
