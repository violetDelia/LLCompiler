function(build_ginac lib_path)
    message("******************************************")
    message("************* build ginac ***************")
    message("******************************************")
    add_subdirectory(${lib_path})
endfunction(build_ginac)