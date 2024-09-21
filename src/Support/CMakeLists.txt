llcompiler_add_library(LLCSupport
    SRC_FILES
    Logger.cpp
    Option.cpp
    Enums.cpp

    LINKS PUBLIC
    LLVMOption

    PRIVATE
    spdlog
)
