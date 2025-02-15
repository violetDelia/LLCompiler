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

#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_

#define LLC_EMITERROR(condistion) \
  CHECK(llc::MLIR, condistion);   \
  if (condistion) return emitOpError()

#define DEBUG_BUILDED_OP(module, op) \
  DEBUG(module) << "create a " << op.getOperationName().str() << "op.";

#define CONVERT_TO_NEW_OP(module, op, new_op)                          \
  rewriter.replaceOp(op, new_op);                                      \
  DEBUG(module) << "convert " << op.getOperationName().str() << " to " \
                << new_op.getOperationName().str() << "."

#define LLC_COMMON_TYPE_CASE(MACRO, type, ...)             \
  MACRO(type.isInteger(1), bool, __VA_ARGS__)              \
  MACRO(type.isSignedInteger(8), int8_t, __VA_ARGS__)      \
  MACRO(type.isSignedInteger(16), int16_t, __VA_ARGS__)    \
  MACRO(type.isSignedInteger(32), int32_t, __VA_ARGS__)    \
  MACRO(type.isSignedInteger(64), int64_t, __VA_ARGS__)    \
  MACRO(type.isSignlessInteger(8), uint8_t, __VA_ARGS__)   \
  MACRO(type.isSignlessInteger(16), uint16_t, __VA_ARGS__) \
  MACRO(type.isSignlessInteger(32), uint32_t, __VA_ARGS__) \
  MACRO(type.isSignlessInteger(64), uint64_t, __VA_ARGS__) \
  MACRO(type.isF32(), float, __VA_ARGS__)                  \
  MACRO(type.isF64(), double, __VA_ARGS__)

#define LLC_RUN_IN_PASS                                                      \
  INFO(llc::MLIR_PASS) << "----- run in pass: " << this->getPassName().str() \
                       << " -----";

#define LLC_RUN_OUT_PASS                                                      \
  INFO(llc::MLIR_PASS) << "----- run out pass: " << this->getPassName().str() \
                       << " -----";

#define LLC_RUN_IN_PATTERN \
  DEBUG(llc::MLIR_PASS) << "run in pattern " << this->getDebugName().str();

#define LLC_RUN_OUT_PATTERN                                          \
  DEBUG(llc::MLIR_PASS) << "rewrite " << op.getOperationName().str() \
                        << " in pattern " << this->getDebugName().str();

#define LLC_ADD_PATTERN_WITH_BENEFIT(PATTERN, benefit, ...) \
  patterns.addWithLabel<PATTERN>({#PATTERN}, context, benefit, ##__VA_ARGS__);

#define LLC_ADD_PATTERN(PATTERN, ...) \
  patterns.addWithLabel<PATTERN>({#PATTERN}, context, ##__VA_ARGS__);

#define LLC_ADD_CONVERSION(PATTERN) \
  patterns.addWithLabel<PATTERN>({#PATTERN}, converter, context);

#define LLC_DEFINR_CONVERSION_PASS(NAME, addPatterns, configTarget,            \
                                   initTypeConverter)                          \
  using namespace mlir::impl;                                                  \
  namespace {                                                                  \
  struct NAME##Pass : NAME##PassBase<NAME##Pass> {                             \
    using NAME##PassBase<NAME##Pass>::NAME##PassBase;                          \
    void runOnOperation() override;                                            \
    void populate##NAME##PassPatterns(TypeConverter& converter,                \
                                      RewritePatternSet& patterns);            \
    void config##NAME##PassTarget(ConversionTarget& target);                   \
    void init##NAME##PassTypeConverter(TypeConverter& converter);              \
  };                                                                           \
  }                                                                            \
                                                                               \
  void NAME##Pass::populate##NAME##PassPatterns(TypeConverter& converter,      \
                                                RewritePatternSet& patterns) { \
    auto context = patterns.getContext();                                      \
    addPatterns                                                                \
  }                                                                            \
                                                                               \
  void NAME##Pass::config##NAME##PassTarget(ConversionTarget& target) {        \
    configTarget                                                               \
  }                                                                            \
                                                                               \
  void NAME##Pass::init##NAME##PassTypeConverter(TypeConverter& converter) {   \
    initTypeConverter                                                          \
  }                                                                            \
                                                                               \
  void NAME##Pass::runOnOperation() {                                          \
    LLC_RUN_IN_PASS                                                            \
    ConversionTarget target(getContext());                                     \
    config##NAME##PassTarget(target);                                          \
    TypeConverter converter;                                                   \
    init##NAME##PassTypeConverter(converter);                                  \
    RewritePatternSet patterns(&getContext());                                 \
    populate##NAME##PassPatterns(converter, patterns);                         \
    if (failed(applyPartialConversion(getOperation(), target,                  \
                                      std::move(patterns))))                   \
      signalPassFailure();                                                     \
    LLC_RUN_OUT_PASS                                                           \
  }

#define LLC_DEFINR_PASS(NAME, addPatterns, beforeApplyPatterns,                \
                        afterApplyPatterns)                                    \
  using namespace mlir::impl;                                                  \
  namespace {                                                                  \
  struct NAME##Pass : NAME##PassBase<NAME##Pass> {                             \
    using NAME##PassBase<NAME##Pass>::NAME##PassBase;                          \
    void runOnOperation() override;                                            \
    void populate##NAME##PassPatterns(RewritePatternSet& patterns);            \
  };                                                                           \
  }                                                                            \
                                                                               \
  void NAME##Pass::populate##NAME##PassPatterns(RewritePatternSet& patterns) { \
    auto context = patterns.getContext();                                      \
    addPatterns                                                                \
  }                                                                            \
                                                                               \
  void NAME##Pass::runOnOperation() {                                          \
    auto* context = &getContext();                                             \
    auto module = getOperation();                                              \
    RewritePatternSet patterns(context);                                       \
    populate##NAME##PassPatterns(patterns);                                    \
    beforeApplyPatterns;                                                       \
    auto config = GreedyRewriteConfig();                                       \
    config.useTopDownTraversal = true;                                         \
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns),       \
                                            config)))                          \
                                                                               \
      signalPassFailure();                                                     \
    afterApplyPatterns;                                                        \
    LLC_RUN_OUT_PASS                                                           \
  }
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_
