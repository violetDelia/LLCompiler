##---------- compiler ----------##
add_executable(compiler "${CMAKE_CURRENT_SOURCE_DIR}/llcompiler.cpp")
target_include_directories(compiler PRIVATE 
    $<BUILD_INTERFACE:${LLCOMPILER_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_DIR}>
)

target_link_libraries(compiler PRIVATE LLCompiler)