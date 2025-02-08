llcompiler_add_library(LLCSupport
    SRC_FILES
    Logger.cpp
    Enums.cpp

    DEPENDS
    LLCEnumIncGen

    LINKS PUBLIC
    LLVMOption

    PRIVATE
    spdlog
)
