include_directories(${CMAKE_SOURCE_DIR}/src/c/pscgen)

add_executable(pscgen_test pscgen_test.c)

target_link_libraries(pscgen_test pscgen_shared)
