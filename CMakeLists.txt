cmake_minimum_required(VERSION 3.26)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

# CMAKE_OSX_ARCHITECTURES must be set before project()
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Target architectures on macOS")

file(READ include/gdstk/gdstk.hpp HEADER_CONTENTS)
string(REGEX MATCH "GDSTK_VERSION \"([^\"]*)\"" _ ${HEADER_CONTENTS})
set(GDSTK_VERSION ${CMAKE_MATCH_1})

project(gdstk
    VERSION ${GDSTK_VERSION}
    LANGUAGES CXX C)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_C_EXTENSIONS OFF)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    include(CTest)
endif()

# Options
option(GDSTK_BUILD_EXAMPLES "Build C++ example programs" OFF)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(GDSTK_BUILD_EXAMPLES ON CACHE BOOL "Build C++ example programs" FORCE)
endif()

add_subdirectory(external)
add_subdirectory(src)

if(GDSTK_BUILD_EXAMPLES)
    add_subdirectory(docs/cpp)
endif()

if("${SKBUILD}" STREQUAL "2")
    add_subdirectory(python)
endif()
