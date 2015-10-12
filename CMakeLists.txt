cmake_minimum_required (VERSION 2.8)
project (pscgen)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

SET(CMAKE_CXX_FLAGS "-std=c++11")

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_MATLAB_BINDINGS "Build Matlab bindings" ON)
#option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
#option(BUILD_DOC "Build documentation" ON)
option(USE_OPENMP "Use OpenMP multi-threading" ON)

add_subdirectory(src)
