llcompiler_add_library(LLCLLVMExtensionTransforms
  SRC_FILES
  AdaptEntryParmsForEngine.cpp

  DEPENDS
  MLIRLLVMExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRIndexDialect
  MLIRPass
)

