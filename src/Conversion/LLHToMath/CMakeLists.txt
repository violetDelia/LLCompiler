llcompiler_add_library(LLCLLHToMath
  SRC_FILES
  LLHToMath.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRArithDialect
  LLCLLHDialect
  MLIRFuncDialect
  MLIRMathDialect
)
