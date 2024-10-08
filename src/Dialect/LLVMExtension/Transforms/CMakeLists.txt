llcompiler_add_library(MLIRLLVMExtensionTransforms
  SRC_FILES
  AdaptEntryParmsForEngine.cpp

  DEPENDS
  MLIRLLVMExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRIndexDialect
  MLIRPass
)

