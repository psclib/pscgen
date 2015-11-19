include_directories(${CMAKE_SOURCE_DIR}/src/c/pscgen)

add_executable(generator_test generator_test.c)

target_link_libraries(generator_test pscgen_shared)
