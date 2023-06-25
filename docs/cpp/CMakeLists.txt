set(ALL_EXAMPLES
    apply_repetition
    first
    flexpaths
    geometry_operations
    merging
    pads
    path_markers
    pcell
    photonics
    polygons
    pos_filtering
    references
    repetitions
    robustpaths
    text
    transforms
    layout
    filtering)

foreach(EXAMPLE ${ALL_EXAMPLES})
    add_executable(${EXAMPLE} EXCLUDE_FROM_ALL "${EXAMPLE}.cpp")
    target_compile_features(${EXAMPLE} PRIVATE cxx_std_11)
    target_link_libraries(${EXAMPLE} gdstk)
    add_test(NAME ${EXAMPLE} COMMAND ${EXAMPLE})
endforeach()

add_custom_target(examples DEPENDS ${ALL_EXAMPLES})
