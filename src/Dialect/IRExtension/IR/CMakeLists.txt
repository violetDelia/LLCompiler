add_mlir_dialect_library(MLIRIRExtension
    Types.cpp
    Dialect.cpp
    Ops.cpp
    Enums.cpp
    Attrs.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/IRExtension/IR

    DEPENDS
    MLIRIRExtensionIncGen

    LINK_LIBS PUBLIC
    MLIRSupport
)
llcompiler_install_mlir_target(MLIRIRExtension DIALECT)