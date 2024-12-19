llcompiler_add_library(LLCMLIRUtility
    SRC_FILES
    Type.cpp
    Builder.cpp
    Attribute.cpp
    File.cpp
    Tool.cpp
    Benefit.cpp

    LINKS PUBLIC
    MLIRParser
    MLIRIR
    LLCSupport
    MLIRLLHDialect

)

llcompiler_add_library(LLCPatternRewriter
    SRC_FILES
    RewritePattern.cpp

    LINKS PUBLIC
    MLIRParser
    MLIRIR
    LLCSupport
)

