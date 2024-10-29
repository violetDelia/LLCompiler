set(LLCOMPILER_TANSFORMLIB_INSTALL_DIR "${LLCOMPILER_INSTALL_INCLUDE_DIR}/TransformLibrary")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryConfig.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/LibraryConfig.h"
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryConfig.install.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/LibraryConfig.install.h"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LibraryConfig.install.h"
    DESTINATION ${LLCOMPILER_TANSFORMLIB_INSTALL_DIR} RENAME LibraryConfig.h)