function(new_policy policy)
    if(POLICY policy)
        cmake_policy(SET policy NEW)
    endif()
endfunction(new_policy)

new_policy(CMP0135)
new_policy(CMP0148)
new_policy(CMP0091)
set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)


