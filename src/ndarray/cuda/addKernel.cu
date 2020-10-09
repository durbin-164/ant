#include "cudaWrapper.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

__global__ void add_(double* a, double* b, double* c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // printf("*****in device: %d*****\n", tid);

  // Boundary check
  if (tid < N) {
    // Each thread adds a single element
    c[tid] = a[tid] + b[tid];
  }
}

namespace cuda
{
      void printList(double * data, int  N){
          double *hostData = (double *)malloc(N);
          cudaMemcpy(hostData, data, N*sizeof(double), cudaMemcpyDeviceToHost);
          for(int i=0; i< N; i++){
            std::cout<<hostData[i]<<" ";
          }
          std::cout<<std::endl;
      }


    ndarray::Array cudaAdd(const ndarray::Array &A, const ndarray::Array &B){
        double *c;

        int N = A.size();

        cudaMalloc(&c, N*sizeof(double));

        int NUM_THREADS = 1 << 10;

        // CTAs per Grid
        // We need to launch at LEAST as many threads as we have elements
        // This equation pads an extra CTA to the grid if N cannot evenly be divided
        // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
        int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

        if(NUM_BLOCKS==1) NUM_THREADS =N;

        // Launch the kernel on the GPU
        // Kernel calls are asynchronous (the CPU program continues execution after
        // call, but no necessarily before the kernel finishes)
        add_<<<NUM_BLOCKS,NUM_THREADS>>>(A.deviceData(), B.deviceData(), c, N);
        cudaDeviceSynchronize();
  
        return ndarray::Array(A.shape(), nullptr, c);
    }
}