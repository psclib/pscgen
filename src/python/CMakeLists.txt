#copies /src/python to build folder
file(COPY . DESTINATION .)

execute_process(COMMAND python setup.py install --user
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
