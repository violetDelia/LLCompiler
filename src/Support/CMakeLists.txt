add_mlir_library(LLCompilerSupport
    Core.cpp
    Logger.cpp
    Option.cpp

    DEPENDS

    LINK_COMPONENTS
    LLVMOption

    LINK_LIBS PUBLIC
    spdlog
)
