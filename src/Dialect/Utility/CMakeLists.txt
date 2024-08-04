llcompiler_add_library(LLCMLIRUtility
    SRC_FILES
    Type.cpp
    Builder.cpp
    Attribute.cpp
    File.cpp

    DEPENDS
    MLIRIR
    LLCSupport
)