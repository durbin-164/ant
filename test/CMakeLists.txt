cmake_minimum_required(VERSION 3.0.0)
# project(ant_test VERSION 0.1.0)

set(TEST_NAME ${PROJECT_NAME}_test)

# set_property(GLOBAL PROPERTY SOURCES_TEST "")

add_subdirectory("ndarray")

get_property(SOURCES_TEST GLOBAL PROPERTY SOURCES_TEST)

# foreach(file ${SOURCES_TEST})
#     message("test file : " ${file})  
# endforeach()


message("executable: " ${TEST_NAME})
add_executable(${TEST_NAME} ${SOURCES_TEST})
# add_library(ant_test ${SOURCES_TEST})
target_link_libraries(${TEST_NAME}  PRIVATE
  gtest
  gtest_main
  ant
  cuda
)

add_test( ${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME} )