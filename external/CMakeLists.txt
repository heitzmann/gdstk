include(GNUInstallDirs)


## CLIPPER ########################################################

set(clipper_HEADERS clipper/clipper.hpp)

set(clipper_SOURCES clipper/clipper.cpp)

add_library(clipper ${clipper_SOURCES} ${clipper_HEADERS})

if(UNIX)
    target_link_libraries(clipper m)
endif(UNIX)

set_target_properties(clipper PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(clipper PRIVATE cxx_std_11)

target_include_directories(clipper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS clipper
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
