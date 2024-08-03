add_mlir_conversion_library(MLIRLLHToTosa
  LLHToTosa.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLCOMPILER_INCLUDE_DIR}/llcompiler/Conversion/LLHToTosa

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTosaDialect
  MLIRLLHDialect
  MLIRPass
)

llcompiler_install_mlir_library(MLIRLLHToTosa CONVERSION)
