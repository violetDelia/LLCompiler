add_mlir_conversion_library(MLIRLLHToTosa
  LLHToTosa.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLCOMPILER_INCLUDE_DIR}/llcompiler/Conversion/LLHToTosa

  DEPENDS
  LLCConversionPassIncGen

  LINK_LIBS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTosaDialect
  MLIRLLHDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
)

llcompiler_install_mlir_library(MLIRLLHToTosa CONVERSION)
