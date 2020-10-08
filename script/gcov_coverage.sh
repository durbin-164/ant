#!/bin/bash
if [ ! -d "build" ] 
then
    mkdir build
else
    rm -r build
    mkdir build
fi

cd build
cmake ..
make ant_test
./test/ant_test
make gcov