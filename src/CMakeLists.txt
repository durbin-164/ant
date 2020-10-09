cmake_minimum_required(VERSION 3.0.0)

add_subdirectory("ndarray")

get_property( SOURCES GLOBAL PROPERTY SOURCES)

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

