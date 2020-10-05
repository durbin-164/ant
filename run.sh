clear
rm -r build
mkdir build
cd build
cmake ..
# # make docs
# find . -executable -type f
make ant_test
./test/ant_test

# rm a.out
# clear
# nvcc -arch=sm_50 main.cpp src/ndarray/core/array.cpp  src/ndarray/cuda/addKernel.cu -I src/ndarray/cuda -I src/ndarray/core
# ./a.out