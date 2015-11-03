set(libsrc pscgen/util.c pscgen/half.c pscgen/nnu.c pscgen/linalg/linalg.c)

add_library(pscgen_shared SHARED ${libsrc})
add_library(pscgen STATIC ${libsrc})

target_link_libraries(pscgen_shared ${OPENBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(pscgen ${OPENBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
