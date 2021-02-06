cmake_minimum_required(VERSION 3.14.0)

file(READ src/gdstk.h HEADER_CONTENTS)
string(REGEX MATCH "GDSTK_VERSION \"([^\"]*)\"" _ ${HEADER_CONTENTS})
set(GDSTK_VERSION ${CMAKE_MATCH_1})

project(gdstk VERSION ${GDSTK_VERSION})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "/nologo /W3 /MD /EHsc /wd4996")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /GL")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -Wformat -Wno-missing-field-initializers -Wno-missing-braces")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

find_package(LAPACK REQUIRED)
foreach(IT ${LAPACK_LIBRARIES})
    set(PRIVATE_LIBS "${PRIVATE_LIBS} ${IT}")
endforeach()

find_package(ZLIB REQUIRED)
foreach(IT ${ZLIB_LIBRARIES})
    set(PRIVATE_LIBS "${PRIVATE_LIBS} ${IT}")
endforeach()

add_library(gdstk STATIC
    src/allocator.cpp
    src/cell.cpp
    src/clipper_tools.cpp
    src/curve.cpp
    src/flexpath.cpp
    src/gdsii.cpp
    src/label.cpp
    src/library.cpp
    src/oasis.cpp
    src/polygon.cpp
    src/property.cpp
    src/rawcell.cpp
    src/reference.cpp
    src/repetition.cpp
    src/robustpath.cpp
    src/style.cpp
    src/utils.cpp
    src/clipperlib/clipper.cpp)

target_include_directories(gdstk PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(gdstk ${LAPACK_LIBRARIES} ${ZLIB_LIBRARIES})

set_target_properties(gdstk PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "src/gdstk.h"
    PRIVATE_HEADER "src/allocator.h;src/array.h;src/cell.h;src/clipper_tools.h;src/curve.h;src/flexpath.h;src/font.h;src/gdsii.h;src/gdswriter.h;src/label.h;src/library.h;src/map.h;src/oasis.h;src/pathcommon.h;src/polygon.h;src/property.h;src/rawcell.h;src/rawsource.h;src/reference.h;src/repetition.h;src/robustpath.h;src/style.h;src/utils.h;src/vec.h")

include(GNUInstallDirs)

install(TARGETS gdstk
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gdstk)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gdstk.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

CONFIGURE_FILE("gdstk.pc.in" "gdstk.pc")
