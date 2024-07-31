llcompiler_add_library(LLCFrontendCore
    SRC_FILES
    Base.cpp
    Builder.cpp
    Importer.cpp
    Option.cpp
    Utility.cpp

    LINKS PUBLIC
    LLVMOption
    LLCSupport
    MLIRTosaDialect
    MLIRLLHDialect
    

    PRIVATE
    onnx
)
