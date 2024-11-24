llcompiler_add_library(MLIRTosaExDialect
    SRC_FILES
    TosaExTypes.cpp
    TosaExDialect.cpp
    TosaExOps.cpp

    DEPENDS
    LLCTosaExDialectIncGen

    LINKS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRFuncDialect
)

# add_mlir_dialect_library(MLIRTosaExDialect
#     TosaExTypes.cpp
#     TosaExDialect.cpp
#     TosaExOps.cpp

#     ADDITIONAL_HEADER_DIRS
#     ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/TosaEx

#     DEPENDS
#     LLCTosaExDialectIncGen

#     LINK_LIBS PUBLIC
#     MLIRIR
#     MLIRInferTypeOpInterface
#     MLIRFuncDialect
# )
# llcompiler_install_mlir_target(MLIRTosaExDialect DIALECT)