cmake_minimum_required(VERSION 3.0.0)
set(TEST_NAME ${PROJECT_NAME}_test)

# set_property(GLOBAL PROPERTY SOURCES_TEST "")

add_subdirectory("ndarray")

get_property(SOURCES_TEST GLOBAL PROPERTY SOURCES_TEST)

add_executable(${TEST_NAME} ${SOURCES_TEST})
# add_library(ant_test ${SOURCES_TEST})
target_link_libraries(${TEST_NAME}  PRIVATE
  gtest
  gtest_main
  ant
  cuda
)


# ***************** For Test Coverage **********************

set(GCOV_OBJECT_DIR 
    ${CMAKE_BINARY_DIR}/src/CMakeFiles/ant.dir
    ${CMAKE_BINARY_DIR}/src/CMakeFiles/ant.dir/ndarray
    ${CMAKE_BINARY_DIR}/src/CMakeFiles/ant.dir/ndarray/core
    ${CMAKE_BINARY_DIR}/src/CMakeFiles/ant.dir/ndarray/cuda
)

set(GCOV_SOURCES_DIR
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/ndarray/*.cpp
  ${CMAKE_SOURCE_DIR}/src/ndarray/core/*.cpp
  ${CMAKE_SOURCE_DIR}/src/ndarray/cuda/*.cu
)

foreach(file ${GCOV_SOURCES_DIR})
  message("File: " ${file})
endforeach()


# Create the gcov target. Run coverage tests with 'make gcov'
add_custom_target(gcov
    COMMAND mkdir -p coverage
    COMMAND ${CMAKE_MAKE_PROGRAM} test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

list(LENGTH GCOV_SOURCES_DIR len1)
math(EXPR len1 "${len1} - 1")


foreach(idx RANGE ${len1})
    list(GET GCOV_SOURCES_DIR ${idx} source)
    list(GET GCOV_OBJECT_DIR ${idx} object)
    
    add_custom_command(TARGET gcov
        COMMAND echo "=================== GCOV ===================="
          COMMAND gcov -b ${source} -o ${object} 
        # COMMAND gcov -b ${GCOV_DIR} -o ${OBJECT_DIR} 
        # COMMAND gcov -b ${CMAKE_SOURCE_DIR}/src/ops/*.cpp  -o ${CMAKE_BINARY_DIR}/src/CMakeFiles/core.dir/ops
        
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/coverage  # Need separate command for this line
    )

endforeach()

add_dependencies(gcov ant_test)

# Make sure to clean up the coverage folder
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES coverage)

add_test( ${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME} )

set_tests_properties(${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION "0;5;10")