if(STATIC_WINDOWS_RUNTIME)
    option(ONNX_USE_MSVC_STATIC_RUNTIME "static" ON)
endif()
add_spdlog("${CMAKE_CURRENT_SOURCE_DIR}/spdlog")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")

#add_glog("${CMAKE_CURRENT_SOURCE_DIR}/glog")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/googletest")

#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/onnx-mlir")