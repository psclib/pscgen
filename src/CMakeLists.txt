add_subdirectory( c )

#if (BUILD_MATLAB_BINDINGS)
#add_subdirectory( matlab )
#endif()

if (BUILD_PYTHON_BINDINGS)
add_subdirectory( python )
endif()

if (BUILD_TESTS)
add_subdirectory( tests )
endif()
