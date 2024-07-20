if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Wall>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w>)
endif()

if(BUILD_LLCOMPILER_TEST)
    build_googletest("${CMAKE_CURRENT_SOURCE_DIR}/googletest")
endif()

build_spdlog("${CMAKE_CURRENT_SOURCE_DIR}/spdlog")
build_onnx("${CMAKE_CURRENT_SOURCE_DIR}/onnx")
build_llvm("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
