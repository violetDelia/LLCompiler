add_mlir_dialect_library(MLIRLLHDialect
    LLHTypes.cpp
    LLHDialect.cpp
    LLHOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH/IR

    DEPENDS
    MLIRLLHDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    LLCSupport
)
llcompiler_install_mlir_target(MLIRLLHDialect DIALECT)