
llcompiler_add_library(MLIRLLHDialect
    SRC_FILES
    LLHTypes.cpp
    LLHDialect.cpp
    LLHOps.cpp
    LLHFolder.cpp
    LLHCast.cpp
    LLHAttrs.cpp
    LLHinfersymbolShape.cpp
    LLHBraodcastableOpInterfaces.cpp
    LLHEnums.cpp
    LLHVerfy.cpp
    LLHCanonicalize.cpp
    LLHSymbolUses

    DEPENDS
    MLIRLLHDialectIncGen

    LINKS PUBLIC
    MLIRIR
    LLCSupport
    MLIRSymbolShapeOpInterfaces
    MLIRBraodcastableOpInterfaces
)

# add_mlir_dialect_library(MLIRLLHDialect
#     LLHTypes.cpp
#     LLHDialect.cpp
#     LLHOps.cpp

#     ADDITIONAL_HEADER_DIRS
#     ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH/IR

#     DEPENDS
#     MLIRLLHDialectIncGen

#     LINK_LIBS PUBLIC
#     MLIRIR
#     LLCSupport
# )
# llcompiler_install_mlir_target(MLIRLLHDialect DIALECT)