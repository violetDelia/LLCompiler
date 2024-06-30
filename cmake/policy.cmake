function(NEW_POLICY policy)
    if(POLICY policy)
        cmake_policy(SET policy NEW)
    endif()
endfunction(NEW_POLICY)

NEW_POLICY(CMP0135)
NEW_POLICY(CMP0148)
NEW_POLICY(CMP0091)
set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)


