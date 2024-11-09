# #---------- compiler ----------##
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(pipeline_depends GLOBAL PROPERTY LLCOMPILER_PIPELINE_DEPENDS)
llcompiler_add_executable(llc-opt
    "${CMAKE_CURRENT_SOURCE_DIR}/llc-opt.cpp"

    ADDITIONAL_INCLUDE_DIRS

    LINKS
    LLCSupport
    LLCPipelines
    MLIROptLib
    LLCompiler 
    ${pipeline_depends}
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)
