#ifndef INCLUDE_LLCOMPILER_CONVERSION_LLHTOHLO_LLHPREPROCESSINGFORHLO_H_
#define INCLUDE_LLCOMPILER_CONVERSION_LLHTOHLO_LLHPREPROCESSINGFORHLO_H_
#include <memory>
namespace mlir {
class MLIRContext;
class TypeConverter;
class Pass;
class RewritePatternSet;
class ConversionTarget;
#define GEN_PASS_DECL_LLHPREPROCESSINGFORHLOPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir
#endif  // INCLUDE_LLCOMPILER_CONVERSION_LLHTOHLO_LLHPREPROCESSINGFORHLO_H_