add_mlir_dialect_library(MLIRIRExtensionEncoding
    Encoding.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llcompiler/IRExtension

    DEPENDS
    MLIRIRExtensionEncodingIncGen

    LINK_LIBS PUBLIC
    MLIRSupport
)
llcompiler_install_mlir_library(MLIRIRExtensionEncoding)