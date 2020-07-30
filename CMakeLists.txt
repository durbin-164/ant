cmake_minimum_required (VERSION 3.0)

project (ant)

enable_language(C)
enable_language(CXX)
set (CMAKE_CXX_STANDARD 14)

#################
#outside packages

#########pybind11
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

message([MAIN] "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
MESSAGE( [Main] " pybind11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
MESSAGE( [Main] " pybind11_LIBRARIES = ${pybind11_LIBRARIES}")

##################################
#####ArrayFire
FIND_PACKAGE(ArrayFire CONFIG REQUIRED)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

message([MAIN] "Found arrayfire v${ArrayFire_VERSION}: ${ArrayFire_INCLUDE_DIRS}")
MESSAGE( [Main] " ArrayFire_INCLUDE_DIRS = ${ArrayFire_INCLUDE_DIRS}")
MESSAGE( [Main] " ArrayFire_LIBRARIES = ${ArrayFire_LIBRARIES}")


##########################################
# Testing enable
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")

enable_testing()
add_subdirectory("googletest")
###################################

#include header file
include_directories("${CMAKE_SOURCE_DIR}/src")

#include subderectory
add_subdirectory("src")
add_subdirectory("tests")

