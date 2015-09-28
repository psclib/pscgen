file(GLOB SOURCES
    pscgen/*.h
    pscgen/*.cpp
)

add_library (pscgen ${SOURCES})
target_link_libraries(pscgen ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

target_include_directories(
  pscgen INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
