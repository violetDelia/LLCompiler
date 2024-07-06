function(add_spdlog lib_path)
    add_subdirectory(${lib_path})
endfunction(add_spdlog)

function(add_onnx lib_path)
    unset(ABSL_ENABLE_INSTALL CACHE)

    if(STATIC_WINDOWS_RUNTIME)
        option(ONNX_USE_MSVC_STATIC_RUNTIME "static" ON)
    endif()

    option(ABSL_ENABLE_INSTALL "" OFF)
    add_subdirectory(${lib_path})
endfunction(add_onnx)

function(add_llvm lib_path)
    unset(BENCHMARK_ENABLE_EXCEPTIONS CACHE)
    unset(BENCHMARK_ENABLE_TESTING CACHE)
    unset(LLVM_BUILD_EXAMPLES CACHE)
    unset(LLVM_ENABLE_ASSERTIONS CACHE)
    unset(LLVM_ENABLE_ASSERTIONS CACHE)
    unset(LLVM_INCLUDE_TESTS CACHE)
    set(LLVM_TARGETS_TO_BUILD "Native;NVPTX;AMDGPU;RISCV"
        CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING
        "Semicolon-separated list of projects to build (${LLVM_KNOWN_PROJECTS}), or \"all\".")
    option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
    option(BENCHMARK_ENABLE_EXCEPTIONS "Enable the use of exceptions in the benchmark library." OFF)
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    option(LLVM_BUILD_EXAMPLES "Build the LLVM example programs. If OFF, just generate build targets." ON)
    option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
    option(LLVM_INCLUDE_TESTS "Generate build targets for the LLVM unit tests." OFF)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
endfunction(add_llvm)
