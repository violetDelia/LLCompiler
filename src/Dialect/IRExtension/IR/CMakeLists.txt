llcompiler_add_library(LLCIRExtension
    SRC_FILES
    Types.cpp
    Dialect.cpp
    Ops.cpp
    Enums.cpp
    Attrs.cpp

    DEPENDS
    LLCIRExtensionIncGen

    LINKS PUBLIC
    MLIRSupport
)
