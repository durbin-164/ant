cmake_minimum_required(VERSION 3.16.0)
project(ant VERSION 0.1.0)

find_package(CUDA REQUIRED)

if (CUDA_FOUND)
   LIST(APPEND CUDA_NVCC_FLAGS "-std=c++14 -arch=sm_50 -rdc=true -Wno-deprecated-gpu-targets")
endif()
enable_language(CUDA)


# enable_language(C)
# enable_language(CXX)
# set (CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -lgtest")
set(CMAKE_CXX_FLAGS "-g -O0 -coverage ")
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)


add_subdirectory("googletest")
enable_testing()





# set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR 
      "${CMAKE_SOURCE_DIR}/src"
      "${CMAKE_SOURCE_DIR}/test"
      "${CMAKE_SOURCE_DIR}/src/ndarray"
      "${CMAKE_SOURCE_DIR}/src/ndarray/core"
      "${CMAKE_SOURCE_DIR}/src/ndarray/cuda"
      "${CMAKE_SOURCE_DIR}/src/ndarray/util"
)

include_directories(${PROJECT_INCLUDE_DIR})

add_subdirectory("src")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory("test")

# Doxygen

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs_doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
