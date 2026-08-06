## Clipper2 (submodule) — built as an OBJECT library, linked directly into gdstk.

set(CLIPPER2_SRC
    Clipper2/CPP/Clipper2Lib/src/clipper.engine.cpp
    Clipper2/CPP/Clipper2Lib/src/clipper.offset.cpp
    Clipper2/CPP/Clipper2Lib/src/clipper.rectclip.cpp
)

add_library(clipper2 OBJECT ${CLIPPER2_SRC})

target_compile_features(clipper2 PRIVATE cxx_std_17)

set_target_properties(clipper2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(clipper2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Clipper2/CPP/Clipper2Lib/include>)
