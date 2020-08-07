cmake_minimum_required (VERSION 3.0)

set(SOURCES 
    math.cpp
) 

# set(INCLUDE_DIR
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     PARENT_SCOPE
# )

add_subdirectory("ops")

add_library(core ${SOURCES})

pybind11_add_module(${PROJECT_NAME} ${SOURCES} bindings.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ArrayFire::afcuda)

