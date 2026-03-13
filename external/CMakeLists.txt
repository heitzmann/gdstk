## Clipper (vendored) — built as an OBJECT library, linked directly into gdstk.

add_library(clipper OBJECT clipper/clipper.cpp clipper/clipper.hpp)

target_compile_features(clipper PRIVATE cxx_std_11)

set_target_properties(clipper PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(clipper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
