llcompiler_add_library(LLCompiler
    SRC_FILES
    Init.cpp
    Utility.cpp

    LINKS
    PUBLIC
    LLCompilerSupport
    PRIVATE
    MLIRIR
)
