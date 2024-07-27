add_mlir_library(MLIRTosaExTransforms
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/TosaEx/Transforms

  DEPENDS
  MLIRTosaExDialectIncGen
  MLIRTosaExPassIncGen

  LINK_LIBS PUBLIC
  MLIRTosaExDialect
  MLIRPass
)
llcompiler_install_mlir_library(MLIRTosaExTransforms TRANSFORM)