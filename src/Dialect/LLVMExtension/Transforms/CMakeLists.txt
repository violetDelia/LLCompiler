llcompiler_add_library(LLCLLVMExtensionTransforms
  SRC_FILES
  AdaptEntryParmsForEngine.cpp

  DEPENDS
  LLCLLVMExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRIndexDialect
  MLIRPass
)

