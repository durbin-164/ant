cmake_minimum_required(VERSION 3.0.0)

get_property( SOURCES GLOBAL PROPERTY SOURCES)

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")


foreach(file ${files})
    list (APPEND SOURCES ${file})
endforeach()

set_property(GLOBAL PROPERTY SOURCES ${SOURCES})


