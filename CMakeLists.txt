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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /nologo /W3 /MD /EHsc /wd4996")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox /GL -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wvla -Wformat -Wno-missing-field-initializers -Wno-missing-braces")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

find_package(ZLIB 1.2.7)
if(NOT ZLIB_FOUND)
    # Check first for header files
    if(DEFINED ENV{ZLIB_INCLUDE_DIRS})
        set(ZLIB_INCLUDE_DIRS $ENV{ZLIB_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "ZLIB headers were not found, either via find_package or env variable")
    endif()

    # Use normal library-finding methods first
    find_library(ZLIB_LIBRARIES zlib)
    # But give precedence to user-set library path
    if(DEFINED ENV{ZLIB_LIBRARIES})
        set(ZLIB_LIBRARIES $ENV{ZLIB_LIBRARIES})
    # This would be set if find_library succedes, e.g. the library is on PATH, but no env-var was set.
    elseif(NOT ZLIB_LIBRARIES)
        message(FATAL_ERROR "ZLIB could not be found, either via find_package, find_library, or env variable.")
    endif()
endif(NOT ZLIB_FOUND)

foreach(IT ${ZLIB_LIBRARIES})
    set(PRIVATE_LIBS "${PRIVATE_LIBS} ${IT}")
endforeach()

add_library(gdstk STATIC
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
    src/clipperlib/clipper.cpp
    src/libqhull_r/global_r.c
    src/libqhull_r/stat_r.c
    src/libqhull_r/geom2_r.c
    src/libqhull_r/poly2_r.c
    src/libqhull_r/merge_r.c
    src/libqhull_r/libqhull_r.c
    src/libqhull_r/geom_r.c
    src/libqhull_r/poly_r.c
    src/libqhull_r/qset_r.c
    src/libqhull_r/mem_r.c
    src/libqhull_r/random_r.c
    src/libqhull_r/usermem_r.c
    src/libqhull_r/userprintf_r.c
    src/libqhull_r/io_r.c
    src/libqhull_r/user_r.c
    src/libqhull_r/rboxlib_r.c
    src/libqhull_r/userprintf_rbox_r.c
)

target_include_directories(gdstk PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(gdstk ${ZLIB_LIBRARIES})

set_target_properties(gdstk PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "src/gdstk.h"
    PRIVATE_HEADER "src/allocator.h;src/array.h;src/cell.h;src/clipper_tools.h;src/curve.h;src/flexpath.h;src/font.h;src/gdsii.h;src/gdswriter.h;src/label.h;src/library.h;src/map.h;src/oasis.h;src/pathcommon.h;src/polygon.h;src/property.h;src/rawcell.h;src/reference.h;src/repetition.h;src/robustpath.h;src/set.h;src/sort.h;src/style.h;src/utils.h;src/vec.h")

include(GNUInstallDirs)

install(TARGETS gdstk
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gdstk)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gdstk.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

CONFIGURE_FILE("gdstk.pc.in" "gdstk.pc")
