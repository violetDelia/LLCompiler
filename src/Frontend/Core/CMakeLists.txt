
llcompiler_add_library(LLCFrontendCore
    SRC_FILES
    Base.cpp
    Importer.cpp
    Option.cpp
    Utility.cpp

    LINKS PUBLIC
    LLCSupport
    LLVMOption

    PRIVATE
    onnx
)
