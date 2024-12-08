llcompiler_add_library(LLCStablehlotoLinalgExtension
  SRC_FILES
  StablehlotoLinalgExtension.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRLinalgDialect
  MLIRLLHDialect
  MLIRFuncDialect
  StablehloOps
)


