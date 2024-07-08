add_mlir_library(LLCLLHPasses
  LLHPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/llcompiler/dialect/LLH/Transforms

  DEPENDS
  MLIRLLHPassesIncGen

  LINK_LIBS PUBLIC
  LLCLLHDialect
)