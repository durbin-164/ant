#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

#include "cudaWrapper.h"
#include "ndarray/cuda/util.cuh"
#include "ndarray/core/broadCasted.h"
#include "ndarray/util/arrayUtil.h"
#include "ndarray/util/util.h"

__global__ void add_(double* A,
                    double* B,
                    double* C,
                    long long N,
                    int *a_stride,
                    int *b_stride,
                    int *cum_shape,
                    int cum_shape_N
                    ) {

    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // printf("tid: %d\n", tid);
    
    int a_index = getIndex(tid, cum_shape, a_stride, cum_shape_N);
    int b_index = getIndex(tid, cum_shape, b_stride, cum_shape_N);
      
    // Boundary check
    if (tid < N) {
      // printf("index: %d %d %lf %lf\n", a_index, b_index, A[a_index], B[b_index]);
      // Each thread adds a single element
      C[tid] = A[a_index] + B[b_index];
    }
}

namespace cuda
{

ndarray::Array cudaAdd(const ndarray::Array &A, const ndarray::Array &B){
        
    ndarray::Shape out_shape = ndarray::getBroadCastedShape(A.shape(), B.shape());

    ndarray::BroadCastedProperty BP = ndarray::getBroadCastedProperty(out_shape, A, B);

    ndarray::Shape cum_mul_shape = ndarray::getCumulativeMultiShape(out_shape);

    ndarray:: LL num_of_element = ndarray::getNumOfElementByShape(out_shape);
  
      
    double *C;
    cudaMalloc(&C, num_of_element*sizeof(double));

    int *a_stride;
    cudaMalloc(&a_stride, BP.a_stride.size()*sizeof(int));
    cudaMemcpy(a_stride, &BP.a_stride[0], BP.a_stride.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    int *b_stride;
    cudaMalloc(&b_stride, BP.b_stride.size()*sizeof(int));
    cudaMemcpy(b_stride, &BP.b_stride[0], BP.b_stride.size()*sizeof(int), cudaMemcpyHostToDevice);

    int *cum_shape;
    cudaMalloc(&cum_shape, cum_mul_shape.size()*sizeof(int));
    cudaMemcpy(cum_shape, &cum_mul_shape[0], cum_mul_shape.size()*sizeof(int), cudaMemcpyHostToDevice);


    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (num_of_element + NUM_THREADS - 1) / NUM_THREADS;

    if(NUM_BLOCKS==1) NUM_THREADS =num_of_element;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    add_<<<NUM_BLOCKS,NUM_THREADS>>>(A.deviceData(),
                                    B.deviceData(),
                                    C,
                                    num_of_element,
                                    a_stride,
                                    b_stride,
                                    cum_shape,
                                    cum_mul_shape.size()
                                      );
    
    cudaFree(a_stride);
    cudaFree(b_stride);
    cudaFree(cum_shape);
    return ndarray::Array(out_shape, nullptr, C);
}

}//end cuda namespace