add_mlir_dialect_library(LLCLLHDialect
    LLHTypes.cpp
    LLHDialect.cpp
    LLHOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH

    DEPENDS
    MLIRLLHOpsIncGen
    MLIRLLHTypesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRFuncDialect
)
