cmake_minimum_required (VERSION 2.8)
project (pscgen)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
#option(BUILD_MATLAB_BINDINGS "Build Matlab bindings" ON)
#option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
#option(BUILD_DOC "Build documentation" ON)

add_subdirectory(src)
