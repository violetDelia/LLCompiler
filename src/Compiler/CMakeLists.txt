llcompiler_add_library(LLCompiler
    SRC_FILES
    Init.cpp
    Utility.cpp

    DEPENDS
    LLCompilerSupport

    LINKS PUBLIC
    MLIRIR
)
