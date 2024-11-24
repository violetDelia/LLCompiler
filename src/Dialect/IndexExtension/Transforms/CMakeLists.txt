llcompiler_add_library(LLCIndexExtensionTransforms
  SRC_FILES
  FoldIndexCast.cpp

  DEPENDS
  LLCIndexExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRIndexDialect
  MLIRPass
)

