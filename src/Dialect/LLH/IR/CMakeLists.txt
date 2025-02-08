llcompiler_add_library(LLCLLHDialect
    SRC_FILES
    LLHTypes.cpp
    LLHDialect.cpp
    LLHOps.cpp
    LLHFolder.cpp
    LLHCast.cpp
    LLHAttrs.cpp
    LLHinfersymbolShape.cpp

    # LLHBraodcastableOpInterfaces.cpp
    LLHEnums.cpp
    LLHVerfy.cpp
    LLHCanonicalize.cpp
    LLHSymbolUses.cpp
    LLHSymbolCanonicalize.cpp
    LLHUtilitys.cpp

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
