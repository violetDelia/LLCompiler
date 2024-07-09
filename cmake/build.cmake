function(llcompiler_add_library TargetName LinkLibs SourceFiles)
    if(NOT ${${SourceFiles}} STREQUAL ${LLCOMPILER_CURRENT_DIR_FILES})
        set(library_source_files ${${SourceFiles}})
    else()
        file(GLOB_RECURSE library_source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    endif()

    # #---------- compile llcimpiler ----------##
    if(BUILD_SHARED_LIBS)
        add_library(${TargetName} SHARED ${library_source_files})
    else()
        add_library(${TargetName} STATIC ${library_source_files})
    endif()

    set_target_properties(${TargetName} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_include_directories(${TargetName}
        PRIVATE
        $<BUILD_INTERFACE:${LLCOMPILER_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_DIR}>
    )
    target_link_libraries(${TargetName} PRIVATE ${${LinkLibs}})
endfunction(llcompiler_add_library)
