cmake_minimum_required(VERSION 3.13.0)
project(gdstk)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-missing-field-initializers -Wshadow")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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
