cmake_minimum_required(VERSION 3.0.0)
project(ant VERSION 0.1.0)


# enable_language(C)
# enable_language(CXX)
# set (CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lgtest")
# set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

add_subdirectory("googletest")
enable_testing()


find_package(CUDA REQUIRED)

if (CUDA_FOUND)
   LIST(APPEND CUDA_NVCC_FLAGS "-std=c++14;-arch=sm_50 ;-arch=compute_50; -g; -G; -Wno-deprecated-gpu-targets")
endif()

enable_language(CUDA)


set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR 
      "${CMAKE_SOURCE_DIR}/src"
      "${CMAKE_SOURCE_DIR}/src/ndarray"
      "${CMAKE_SOURCE_DIR}/src/ndarray/core"
      "${CMAKE_SOURCE_DIR}/src/ndarray/cuda"
)

include_directories(${PROJECT_INCLUDE_DIR})

set(SOURCES 
    ${PROJECT_SOURCE_DIR}/hello.cpp
    ${PROJECT_SOURCE_DIR}/hello.h
    )

set_property(GLOBAL PROPERTY SOURCES ${SOURCES})

add_subdirectory("src")

get_property( SOURCES GLOBAL PROPERTY SOURCES)

# foreach(file ${SOURCES})
#   message("soruce file: " ${file})
# endforeach()



# Set up such that your IDE organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})


# add_library(${PROJECT_NAME} ${SOURCES})
CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED
  ${SOURCES}
  # OPTIONS -arch sm_52
)

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
)

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
