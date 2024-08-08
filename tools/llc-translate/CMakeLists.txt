# #---------- compiler ----------##
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
llcompiler_add_executable(llc-translate
    "${CMAKE_CURRENT_SOURCE_DIR}/llc-translate.cpp"

    LINKS
    ${dialect_libs}
    ${translation_libs}
    LLVMSupport
    LLCompiler
    MLIRPass
    MLIRLLHToTosa
    MLIRIR
    MLIRParser
    MLIRSPIRVDialect
    MLIRTranslateLib
    MLIRSupport
)
