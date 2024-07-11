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
    target_link_libraries(${TargetName} PUBLIC ${${LinkLibs}})
endfunction(llcompiler_add_library)

function(add_catch_test)
    set(options)
    set(oneValueArgs NAME COST)
    set(multiValueArgs LABELS DEPENDS REFERENCE_FILES)
    cmake_parse_arguments(add_catch_test
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "defining a test ...")
    message(STATUS " NAME: ${add_catch_test_NAME}")
    message(STATUS " LABELS: ${add_catch_test_LABELS}")
    message(STATUS " COST: ${add_catch_test_COST}")
    message(STATUS " REFERENCE_FILES: ${add_catch_test_REFERENCE_FILES}")

    
endfunction()
