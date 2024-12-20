function(build_symengine lib_path)
    message("******************************************")
    message("************* build symengine ***************")
    message("******************************************")
    set(WITH_SYMENGINE_ASSERT ON)
    set(WITH_SYMENGINE_RCP ON)
    set(WITH_SYMENGINE_THREAD_SAFE ON)
    set(WITH_ECM OFF)
    set(WITH_PRIMESIEVE OFF)
    set(WITH_FLINT OFF)
    set(WITH_ARB OFF)
    set(WITH_TCMALLOC OFF)
    set(WITH_PIRANHA OFF)
    set(WITH_MPFR OFF)
    set(WITH_MPC OFF)
    set(WITH_LLVM OFF)
    set(WITH_OPENMP OFF)
    set(WITH_SYSTEM_CEREAL OFF)
    set(BUILD_TESTS OFF)
    add_subdirectory(${lib_path})
endfunction(build_symengine)