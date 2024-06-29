
function(add_spdlog lib_path)
    add_subdirectory(${lib_path})
endfunction(add_spdlog)

function(NEW_POLICY policy)
    if(POLICY policy)
        cmake_policy(SET policy NEW)
    endif()
endfunction(NEW_POLICY)
