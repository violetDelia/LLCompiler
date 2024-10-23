llcompiler_add_library(MLIRIRExtension
    SRC_FILES
    Types.cpp
    Dialect.cpp
    Ops.cpp
    Enums.cpp
    Attrs.cpp

    DEPENDS
    MLIRIRExtensionIncGen

    LINKS PUBLIC
    MLIRSupport
)
