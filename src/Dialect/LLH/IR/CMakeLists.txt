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
    LLHCanonicalizeIncGen

    LINKS PUBLIC
    MLIRIR
    LLCSupport
    MLIRSymbolShapeOpInterfaces
    MLIRBraodcastableOpInterfaces
)

