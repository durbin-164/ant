cmake_minimum_required(VERSION 3.0)
project(ant_test)

set(TESTS
  math_test.cpp
)

add_subdirectory("ops")


MESSAGE( [Main] " Test sources = ${TESTS}")

add_executable(${PROJECT_NAME} ${TESTS})
add_library(core_test ${TESTS})
target_link_libraries(${PROJECT_NAME} PRIVATE
  gtest
  gtest_main
  core
  ArrayFire::afcpu
)


#################For test coverage

set(OBJECT_DIR 
    ${CMAKE_BINARY_DIR}/src/CMakeFiles/core.dir
)
message("-- Object files will be output to: ${OBJECT_DIR}")

# Create the gcov target. Run coverage tests with 'make gcov'
add_custom_target(gcov
    COMMAND mkdir -p coverage
    COMMAND ${CMAKE_MAKE_PROGRAM} test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set(GCOV_DIR
  ${CMAKE_SOURCE_DIR}/src/*.cpp
)

message("---GCOV DIR : ${GCOV_DIR}")

add_custom_command(TARGET gcov
    COMMAND echo "=================== GCOV ===================="
    COMMAND gcov -b ${GCOV_DIR} -o ${OBJECT_DIR} 
    COMMAND gcov -b ${CMAKE_SOURCE_DIR}/src/ops/*.cpp  -o ${CMAKE_BINARY_DIR}/src/CMakeFiles/core.dir/ops
    #     | grep -A 5 "Adder.cpp" > CoverageSummary.tmp
    # COMMAND cat CoverageSummary.tmp
    # COMMAND echo "-- Coverage files have been output to ${CMAKE_BINARY_DIR}/coverage"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/coverage  # Need separate command for this line
    )

add_dependencies(gcov ant_test)

# Make sure to clean up the coverage folder
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES coverage)

# Create the gcov-clean target. This cleans the build as well as generated 
# .gcda and .gcno files.
# add_custom_target(scrub
#     COMMAND ${CMAKE_MAKE_PROGRAM} clean
#     COMMAND rm -f ${OBJECT_DIR}/*.gcno
#     COMMAND rm -f ${OBJECT_DIR}/*.gcda
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     )

add_test(output_test ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})
set_tests_properties(output_test PROPERTIES PASS_REGULAR_EXPRESSION "0;5;10")
