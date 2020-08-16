cmake_minimum_required(VERSION 3.13.0)
project(gdstk)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "/nologo /W3 /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /GL")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -Wformat -Wno-missing-field-initializers -Wno-missing-braces")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

get_property(IS_MULTICONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTICONFIG)
    file(WRITE ${CMAKE_BINARY_DIR}/is_multiconfig "TRUE")
endif()

find_package(LAPACK REQUIRED)
file(WRITE ${CMAKE_BINARY_DIR}/lapack_libs "")
foreach(IT ${LAPACK_LIBRARIES})
    file(APPEND ${CMAKE_BINARY_DIR}/lapack_libs " ${IT}")
endforeach()

add_library(gdstk STATIC
    src/cell.cpp
    src/clipperlib
    src/clipper_tools.cpp
    src/curve.cpp
    src/flexpath.cpp
    src/label.cpp
    src/library.cpp
    src/polygon.cpp
    src/property.cpp
    src/rawcell.cpp
    src/reference.cpp
    src/robustpath.cpp
    src/style.cpp
    src/utils.cpp
    src/clipperlib/clipper.cpp
    )
target_link_libraries(gdstk ${LAPACK_LIBRARIES})
set_property(TARGET gdstk PROPERTY POSITION_INDEPENDENT_CODE ON)
