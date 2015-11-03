find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/c/pscgen)

add_library(pypscgen SHARED pypscgen.c)
target_link_libraries(pypscgen ${PYTHON_LIBRARY} pscgen_shared)
#set_target_properties(pypscgen PROPERTIES PREFIX "" SUFFIX ".so")
