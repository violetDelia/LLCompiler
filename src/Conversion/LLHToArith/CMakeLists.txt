llcompiler_add_library(LLCLLHToArith
  SRC_FILES
  LLHToArith.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRArithDialect
  LLCLLHDialect
  MLIRFuncDialect
)
