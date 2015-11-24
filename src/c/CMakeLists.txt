#copies /src/c to build folder
file(COPY . DESTINATION .)

set(libsrc pscgen/util.c pscgen/nnu_storage.c pscgen/nnu_dict.c
    pscgen/linalg/linalg.c pscgen/nnu_generator.c)

add_library(pscgen_shared SHARED ${libsrc})
add_library(pscgen STATIC ${libsrc})

target_link_libraries(pscgen_shared ${OPENBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(pscgen ${OPENBLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_definitions ("-pedantic -Wall -Wextra -Wwrite-strings
                  -Wno-unknown-pragmas")
set(CMAKE_BUILD_TYPE Release)
