llcompiler_add_library(MLIRLLHToArith
  SRC_FILES
  LLHToArith.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRArithDialect
  MLIRLLHDialect
  MLIRFuncDialect
)
