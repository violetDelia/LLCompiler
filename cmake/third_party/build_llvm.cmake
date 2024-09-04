function(build_llvm lib_path)
    message("******************************************")
    message("************* build llvm *****************")
    message("******************************************")
    set(LLVM_TARGETS_TO_BUILD
        #AArch64
        #AMDGPU
        ARM
        AVR
        BPF
        Hexagon
        Lanai
        LoongArch
        Mips
        MSP430
        NVPTX
        PowerPC
        RISCV
        #Sparc
        SystemZ
        VE
        WebAssembly
        X86
        XCore)
    set(LLVM_ENABLE_PROJECTS "mlir")
    set(LLVM_ENABLE_RTTI OFF)
    set(LLVM_ENABLE_ASSERTIONS ON)
    set(LLVM_INCLUDE_TESTS ON)
    set(LLVM_BUILD_EXAMPLES OFF)
    set(LLVM_BUILD_TOOLS ON)
    set(LLVM_INCLUDE_BENCHMARKS OFF)
    set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)
    set(MLIR_ENABLE_BINDINGS_PYTHON ON)
    add_subdirectory(${lib_path})
    execute_process(COMMAND "export PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir/python_packages/mlir_core")
    message(STATUS "export PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir/python_packages/mlir_core")
endfunction(build_llvm)