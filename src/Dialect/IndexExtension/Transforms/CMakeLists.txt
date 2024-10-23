llcompiler_add_library(MLIRIndexExtensionTransforms
  SRC_FILES
  FoldIndexCast.cpp

  DEPENDS
  MLIRIndexExtensionPassIncGen
  

  LINKS PUBLIC
  MLIRIndexDialect
  MLIRPass
)

