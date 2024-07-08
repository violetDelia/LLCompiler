add_mlir_library(LLCLLHPasses
  LLHPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH/Transforms

  DEPENDS
  MLIRLLHPassesIncGen

  LINK_LIBS PUBLIC
  LLCLLHDialect
)