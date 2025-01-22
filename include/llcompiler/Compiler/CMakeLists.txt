
if(DEVELOPER_MODE)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ToolPath.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/ToolPath.h"
    )
else()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ToolPath.install.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/ToolPath.h"
    )
endif()
