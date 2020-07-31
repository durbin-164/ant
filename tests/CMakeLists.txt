cmake_minimum_required(VERSION 3.0)
project(ant_test)

set(TESTS
  math_test.cpp
)

add_executable(${PROJECT_NAME} ${TESTS})
add_library(core_test ${TESTS})
target_link_libraries(${PROJECT_NAME} PRIVATE
  gtest
  gtest_main
  core
  ArrayFire::afcpu
)


