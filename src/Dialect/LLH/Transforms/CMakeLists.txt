add_mlir_library(MLIRLLHTransforms
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH/Transforms

  DEPENDS
  MLIRLLHDialectIncGen
  MLIRLLHPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLHDialect
  MLIRPass
)
llcompiler_install_mlir_library(MLIRLLHTransforms TRANSFORM)