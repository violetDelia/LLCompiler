llcompiler_add_library(LLCMLIRUtility
    SRC_FILES
    Type.cpp
    Builder.cpp
    Attribute.cpp
    File.cpp
    Tool.cpp

    LINKS PUBLIC
    MLIRParser
    MLIRIR
    LLCSupport
)
