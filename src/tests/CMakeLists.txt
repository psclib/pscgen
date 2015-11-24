include_directories(${CMAKE_SOURCE_DIR}/src/c/pscgen)

add_executable(generator_test generator_test.c)
add_executable(pscgen_test pscgen_test.c)
add_executable(classifier_test classifier_test.c)

target_link_libraries(generator_test pscgen_shared)
target_link_libraries(pscgen_test pscgen_shared)
target_link_libraries(classifier_test pscgen_shared)
