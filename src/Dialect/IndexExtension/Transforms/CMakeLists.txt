llcompiler_add_library(LLCIndexExtensionTransforms
  SRC_FILES
  FoldIndexCast.cpp

  DEPENDS
  MLIRIndexExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRIndexDialect
  MLIRPass
)

