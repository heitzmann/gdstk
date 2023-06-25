## CLIPPER ########################################################

set(clipper_HEADERS clipper/clipper.hpp)

set(clipper_SOURCES clipper/clipper.cpp ${clipper_HEADERS})

add_library(clipper ${clipper_SOURCES})

if(UNIX)
    target_link_libraries(clipper m)
endif(UNIX)

set_target_properties(clipper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "${clipper_HEADERS}")

include(GNUInstallDirs)
install(TARGETS clipper
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/clipper)


## QHULL ##########################################################

set(libqhullr_HEADERS
    qhull/src/libqhull_r/libqhull_r.h
    qhull/src/libqhull_r/geom_r.h
    qhull/src/libqhull_r/io_r.h
    qhull/src/libqhull_r/mem_r.h
    qhull/src/libqhull_r/merge_r.h
    qhull/src/libqhull_r/poly_r.h
    qhull/src/libqhull_r/qhull_ra.h
    qhull/src/libqhull_r/qset_r.h
    qhull/src/libqhull_r/random_r.h
    qhull/src/libqhull_r/stat_r.h
    qhull/src/libqhull_r/user_r.h)

set(libqhullr_SOURCES
    qhull/src/libqhull_r/global_r.c
    qhull/src/libqhull_r/stat_r.c
    qhull/src/libqhull_r/geom2_r.c
    qhull/src/libqhull_r/poly2_r.c
    qhull/src/libqhull_r/merge_r.c
    qhull/src/libqhull_r/libqhull_r.c
    qhull/src/libqhull_r/geom_r.c
    qhull/src/libqhull_r/poly_r.c
    qhull/src/libqhull_r/qset_r.c
    qhull/src/libqhull_r/mem_r.c
    qhull/src/libqhull_r/random_r.c
    qhull/src/libqhull_r/usermem_r.c
    qhull/src/libqhull_r/userprintf_r.c
    qhull/src/libqhull_r/io_r.c
    qhull/src/libqhull_r/user_r.c
    qhull/src/libqhull_r/rboxlib_r.c
    qhull/src/libqhull_r/userprintf_rbox_r.c
    ${libqhullr_HEADERS})

add_library(qhullstatic_r ${libqhullr_SOURCES})

if(UNIX)
    target_link_libraries(qhullstatic_r m)
endif(UNIX)

set_target_properties(qhullstatic_r PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "${libqhullr_HEADERS}")

include(GNUInstallDirs)
install(TARGETS qhullstatic_r
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libqhull_r)
