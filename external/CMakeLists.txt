## CLIPPER ########################################################

set(clipper_HEADERS clipper/clipper.hpp)

set(clipper_SOURCES clipper/clipper.cpp ${clipper_HEADERS})

add_library(clipper ${clipper_SOURCES})

if(UNIX)
    target_link_libraries(clipper m)
endif(UNIX)

set_target_properties(clipper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "${clipper_HEADERS}")

include(GNUInstallDirs)
install(TARGETS clipper
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/clipper)
