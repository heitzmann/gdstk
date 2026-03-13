include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# --- Dependencies ---

find_package(ZLIB 1.2.7 REQUIRED)
find_package(Qhull 8 REQUIRED)

# --- Source and header lists ---

set(HEADER_LIST
    "${gdstk_SOURCE_DIR}/include/gdstk/allocator.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/array.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/cell.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/clipper_tools.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/curve.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/flexpath.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/font.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/gdsii.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/gdstk.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/gdswriter.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/label.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/layername.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/library.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/map.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/oasis.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/pathcommon.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/polygon.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/property.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/raithdata.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/rawcell.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/reference.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/repetition.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/robustpath.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/set.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/sort.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/style.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/tagmap.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/utils.hpp"
    "${gdstk_SOURCE_DIR}/include/gdstk/vec.hpp")

set(SOURCE_LIST
    cell.cpp
    clipper_tools.cpp
    curve.cpp
    flexpath.cpp
    gdsii.cpp
    label.cpp
    layername.cpp
    library.cpp
    oasis.cpp
    polygon.cpp
    property.cpp
    raithdata.cpp
    rawcell.cpp
    reference.cpp
    repetition.cpp
    robustpath.cpp
    style.cpp
    utils.cpp)

# --- Library target ---

add_library(gdstk STATIC ${SOURCE_LIST} ${HEADER_LIST} $<TARGET_OBJECTS:clipper>)
add_library(gdstk::gdstk ALIAS gdstk)

target_compile_features(gdstk PUBLIC cxx_std_11)

set_target_properties(gdstk PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "${HEADER_LIST}")

# Compiler warnings (applied only to this target, not propagated to consumers)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(gdstk PRIVATE /nologo /W3 /EHsc /wd4996)
else()
    target_compile_options(gdstk PRIVATE
        -Wall -Wextra -Wshadow -Wvla -Wformat
        -Wno-missing-field-initializers -Wno-missing-braces
        -Wno-cast-function-type -Wno-unused-parameter)
endif()

if(APPLE)
    set_target_properties(gdstk PROPERTIES
        OSX_DEPLOYMENT_TARGET "10.9")
endif()

target_include_directories(gdstk
    PUBLIC
        $<BUILD_INTERFACE:${gdstk_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        $<TARGET_PROPERTY:clipper,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(gdstk
    PUBLIC
        ZLIB::ZLIB)

# Qhull is only used internally (src/utils.cpp) — no Qhull headers are exposed
# in gdstk's public API. We use BUILD_LOCAL_INTERFACE to link Qhull during the
# build without it appearing in the exported targets file. GdstkConfig.cmake
# handles finding Qhull and appending it to gdstk::gdstk's link interface for
# downstream static-library consumers.
target_link_libraries(gdstk PRIVATE $<BUILD_LOCAL_INTERFACE:QHULL::QHULL>)

if(UNIX)
    target_link_libraries(gdstk PUBLIC m)
endif()

# --- IDE source grouping ---

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

# --- Install rules ---

install(TARGETS gdstk
    EXPORT gdstkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gdstk)

install(EXPORT gdstkTargets
    FILE gdstkTargets.cmake
    NAMESPACE gdstk::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gdstk)

# --- Package config files ---

configure_package_config_file(
    "${gdstk_SOURCE_DIR}/cmake/gdstkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gdstkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gdstk)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gdstkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gdstkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gdstkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gdstk)

# Install FindQhull.cmake so downstream consumers can find Qhull
install(FILES
    "${gdstk_SOURCE_DIR}/cmake_modules/FindQhull.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cmake_modules)

# --- pkg-config ---

configure_file(
    "${gdstk_SOURCE_DIR}/cmake/gdstk.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gdstk.pc"
    @ONLY)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gdstk.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
