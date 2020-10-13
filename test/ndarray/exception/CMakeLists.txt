cmake_minimum_required(VERSION 3.0.0)

get_property( SOURCES_TEST GLOBAL PROPERTY SOURCES_TEST)

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")


foreach(file ${files})
    list (APPEND SOURCES_TEST ${file})
endforeach()

set_property(GLOBAL PROPERTY SOURCES_TEST ${SOURCES_TEST})
