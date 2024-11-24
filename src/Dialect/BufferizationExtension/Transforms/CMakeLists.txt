llcompiler_add_library(LLCBufferizationExtensionTransforms
  SRC_FILES
  AllocToArg.cpp

  DEPENDS
  LLCBufferizationExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRFuncDialect
  MLIRBufferizationDialect
  MLIRPass
)


