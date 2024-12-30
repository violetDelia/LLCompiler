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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_LOADWEIGHTPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

void parseNpyInfo(FILE* fp, mlir::SmallVector<int64_t>& shape,
                  std::string& type_info, bool& fortran_order,
                  size_t& data_bytes) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  CHECK_EQ(llc::MLIR, res, 11) << "parse_npy_header: failed fread";
  std::string header = fgets(buffer, 256, fp);
  CHECK_EQ(llc::MLIR, header[header.size() - 1], '\n');
  size_t loc1, loc2;
  // fortran order
  loc1 = header.find("'fortran_order'");
  CHECK_NE(llc::MLIR, loc1, std::string::npos)
      << "parse_npy_header: failed to find header keyword: 'fortran_order'";
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True" ? true : false);
  // shape
  loc1 = header.find("(");
  CHECK_NE(llc::MLIR, loc1, std::string::npos)
      << "parse_npy_header: failed to find header keyword: '(' or ')'";
  loc2 = header.find(")");
  CHECK_NE(llc::MLIR, loc2, std::string::npos)
      << "parse_npy_header: failed to find header keyword: '(' or ')'";
  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  shape.clear();
  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  size_t size{1};
  while (std::regex_search(str_shape, sm, num_regex)) {
    auto dim = std::stoi(sm[0].str());
    shape.push_back(dim);
    size *= dim;
    str_shape = sm.suffix().str();
  }
  // type info
  loc1 = header.find("descr");
  CHECK_NE(llc::MLIR, loc1, std::string::npos)
      << "parse_npy_header: failed to find header keyword: 'descr'";
  loc1 += 9;
  bool littleEndian =
      (header[loc1] == '<' || header[loc1] == '|' ? true : false);
  assert(littleEndian);
  type_info = header.substr(loc1 + 1, 2);
  loc2 = type_info.find("'");
  data_bytes = atoi(type_info.substr(1, loc2).c_str());
  data_bytes *= size;
}

Type npyTypeInfoToType(const char* type_info, Builder* build) {
  LLC_COMPARE_AND_RETURN(type_info, "f4", build->getF32Type())
  UNIMPLEMENTED(llc::MLIR) << " convert:" << type_info;
}

#define DENSE_ATTR(Type)                                        \
  mlir::DenseElementsAttr::get(                                 \
      type, llvm::ArrayRef<Type>(reinterpret_cast<Type*>(data), \
                                 reinterpret_cast<Type*>(data + data_bytes)))

mlir::DenseElementsAttr loadNpyFile(mlir::ShapedType type,
                                    const llvm::StringRef& file,
                                    Builder* build) {
  FILE* fp = fopen(file.str().c_str(), "rb");
  CHECK(llc::MLIR, fp) << "read file error: " << file.str();
  mlir::SmallVector<int64_t> shape;
  std::string type_info;
  size_t data_bytes;
  bool fortran_order;
  parseNpyInfo(fp, shape, type_info, fortran_order, data_bytes);
  char* data = reinterpret_cast<char*>(malloc(data_bytes));
  size_t nread = fread(data, 1, data_bytes, fp);
  fclose(fp);
  CHECK_EQ(llc::MLIR, nread, data_bytes) << "read failed";
  LLC_COMPARE_AND_RETURN(type_info.c_str(), "f4", DENSE_ATTR(float))
  LLC_COMPARE_AND_RETURN(type_info.c_str(), "i8", DENSE_ATTR(int64_t))
  UNIMPLEMENTED(llc::MLIR) << type_info.c_str();
}
#undef DENSE_ATTR

mlir::DenseElementsAttr loadWeightFile(mlir::ShapedType type,
                                       const llvm::StringRef& file,
                                       Builder* build) {
  if (file.ends_with(".npy")) {
    return loadNpyFile(type, file, build);
  }
  UNIMPLEMENTED(llc::MLIR);
  return mlir::DenseElementsAttr();
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct LoadWeightOp : public LLHOpRewritePattern<WeightOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(WeightOp op) const final { return llvm::success(); }
  void rewrite(WeightOp op, LLHPatternRewriter& rewriter) const final {
    auto weight_file = op.getWeightFile();
    auto type = op->getResult(0).getType();
    CHECK(llc::MLIR, isa<ShapedType>(type));
    auto shape = mlir::cast_or_null<ShapedType>(type);
    auto tensor =
        RankedTensorType::get(shape.getShape(), shape.getElementType());
    INFO(llc::DEBUG) << weight_file.str();
    auto value = loadWeightFile(tensor, weight_file, &rewriter);
    auto const_op = rewriter.create<llh::ConstantOp>(op->getLoc(), value);
    std::cout << std::endl << std::endl;
    rewriter.replaceOp(op, const_op);
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateLoadWeightPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.insert<LoadWeightOp>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct LoadWeightPass : llh::impl::LoadWeightPassBase<LoadWeightPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void LoadWeightPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  RewritePatternSet patterns(context);
  populateLoadWeightPassPatterns(patterns);
  auto op = getOperation();
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
