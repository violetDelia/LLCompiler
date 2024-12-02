set(LLVM_TARGET_DEFINITIONS LLHCanonicalize.td)
mlir_tablegen(LLHCanonicalize.inc -gen-rewriters)
add_public_tablegen_target(LLHCanonicalizeIncGen)

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
    LLHSymbolUses.cpp

    ADDITIONAL_INCLUDE_DIRS
    ${SYMENGINE_INCLUDE_DIRS}

    DEPENDS
    LLCLLHDialectIncGen
    LLHCanonicalizeIncGen

    LINKS PUBLIC
    MLIRIR
    LLCSupport
    MLIRSymbolShapeOpInterfaces
    MLIRBraodcastableOpInterfaces
    symengine
)
