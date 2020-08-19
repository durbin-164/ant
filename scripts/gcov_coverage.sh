#!/bin/bash
if [ ! -d "build" ] 
then
    mkdir build
else
    rm -r build
    mkdir build
fi

cd build
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 ..
make gcov