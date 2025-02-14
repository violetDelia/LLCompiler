function(build_spdlog lib_path)
    message("******************************************")
    message("************* build spdlog ***************")
    message("******************************************")

    if(BUILD_SHARED_LIBS)
        set(SPDLOG_BUILD_SHARED ON)
    else()
        set(SPDLOG_BUILD_SHARED OFF)
    endif()
    set(SPDLOG_USE_STD_FORMAT OFF)
    set(SPDLOG_BUILD_TESTS OFF)
    set(SPDLOG_INSTALL ON)
    set(SPDLOG_FMT_EXTERNAL OFF)
    add_subdirectory(${lib_path})
endfunction(build_spdlog)