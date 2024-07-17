# function(llcompiler_add_library name)
# cmake_parse_arguments(ARG
# ""
# ""
# "SRC_FILES;DEPENDS;LINKS"
# ${ARGN}
# )

# # #---------- compile lib ----------##
# if(NOT ARG_SRC_FILES)
# message(FATAL_ERROR "No source files specified for library ${name}")
# endif()

# if(BUILD_SHARED_LIBS)
# add_library(${name} SHARED ${ARG_SRC_FILES})
# else()
# add_library(${name} STATIC ${ARG_SRC_FILES})
# endif()

# target_include_directories(${name}
# PRIVATE
# $<BUILD_INTERFACE:${LLCOMPILER_INCLUDE_DIR}>
# $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_DIR}>
# )

# set_target_properties(${name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

# if(ARG_DEPENDS)
# add_dependencies(${name} ${ARG_DEPENDS})
# endif()

# target_link_libraries(${name} PUBLIC ${ARG_LINKS})
# endfunction()
