# function(build_utf8_range lib_path)
# message("build utf8_range")
# set(utf8_range_ENABLE_TESTS OFF)
# set(utf8_range_ENABLE_INSTALL ON)
# add_subdirectory(${lib_path})
# endfunction(build_utf8_range)
function(build_abseil lib_path)
    message("******************************************")
    message("************* build abseil ***************")
    message("******************************************")
    set(ABSL_PROPAGATE_CXX_STD ON)
    set(ABSL_ENABLE_INSTALL ON)

    if(BUILD_SHARED_LIBS)
        set(ABSL_BUILD_DLL ON)
        set(ABSL_BUILD_MONOLITHIC_SHARED_LIBS ON)
    else()
        set(ABSL_BUILD_DLL OFF)
        set(ABSL_BUILD_MONOLITHIC_SHARED_LIBS OFF)
    endif()

    set(ABSL_BUILD_TESTING OFF)
    set(ABSL_BUILD_TEST_HELPERS OFF)
    add_subdirectory(${lib_path})
endfunction(build_abseil)

# function(build_protobuf lib_path)
# message("build protobuf")
# set(protobuf_INSTALL OFF)
# set(protobuf_BUILD_TESTS OFF)
# set(protobuf_BUILD_CONFORMANCE OFF)
# set(protobuf_BUILD_EXAMPLES OFF)
# set(protobuf_BUILD_LIBUPB OFF)
# add_subdirectory(${lib_path})
# endfunction(build_protobuf)
function(build_onnx lib_path)
    message("******************************************")
    message("************* build onnx *****************")
    message("******************************************")

    if(STATIC_WINDOWS_RUNTIME)
        set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
    endif()

    if(LLCOMPILER_BUILD_WITH_ONNX_ML)
        set(ONNX_ML ON)
    else()
        set(ONNX_ML OFF)
    endif()

    set(ONNX_BUILD_SHARED_LIBS OFF)
    set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF)

    set(ONNX_DISABLE_EXCEPTIONS ON)
    set(ONNX_VERIFY_PROTO3 ON)
    set(BUILD_ONNX_PYTHON OFF)
    set(ONNX_BUILD_TESTS OFF)
    set(ONNX_WERROR OFF)
    set(ONNX_DISABLE_STATIC_REGISTRATION ON)
    add_subdirectory(${lib_path})
endfunction(build_onnx)

function(build_llvm lib_path)
    message("******************************************")
    message("************* build llvm *****************")
    message("******************************************")
    set(LLVM_TARGETS_TO_BUILD "all")
    set(LLVM_ENABLE_PROJECTS "mlir")
    set(MLIR_ENABLE_BINDINGS_PYTHON ON)
    set(LLVM_ENABLE_RTTI OFF)
    set(LLVM_ENABLE_ASSERTIONS ON)
    set(LLVM_INCLUDE_TESTS OFF)
    set(LLVM_BUILD_EXAMPLES OFF)
    set(LLVM_INCLUDE_BENCHMARKS OFF)
    set(LLVM_INSTALL_UTILS ON)
    set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
endfunction(build_llvm)

function(build_googletest lib_path)
    message("******************************************")
    message("************* build googletest ***************")
    message("******************************************")
    set(INSTALL_GTEST OFF)
    set(gtest_build_tests OFF)
    set(gtest_build_samples OFF)
    set(gmock_build_tests OFF)
    add_subdirectory(${lib_path})
endfunction(build_googletest)

function(build_spdlog lib_path)
    message("******************************************")
    message("************* build spdlog ***************")
    message("******************************************")

    if(BUILD_SHARED_LIBS)
        set(SPDLOG_BUILD_SHARED ON)
    else()
        set(SPDLOG_BUILD_SHARED OFF)
    endif()

    set(SPDLOG_BUILD_TESTS OFF)
    set(SPDLOG_INSTALL ON)
    add_subdirectory(${lib_path})
endfunction(build_spdlog)
