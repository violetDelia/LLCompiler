add_mlir_library(LLCompiler
    Init.cpp
    Utility.cpp

    DEPENDS
    LLCompilerSupport

    LINK_COMPONENTS

    LINK_LIBS PUBLIC
    MLIRIR
)
