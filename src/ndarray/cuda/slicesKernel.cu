#include "cudaWrapper.h"
#include "ndarray/cuda/util.cuh"
#include "ndarray/util/arrayUtil.h"
#include "ndarray/core/slices.h"
#include "ndarray/core/dataType.h"
#include <iostream>

__global__ void slice_(
                     double *A,
                     double *B,
                     long long N,
                     int *stride,
                     int *cum_shape,
                     int cum_shape_N,
                     int start_index
                )
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // printf("tid: %d\n", tid);
    
    int index = getIndex(tid, cum_shape, stride, cum_shape_N, start_index);
      
    // Boundary check
    if (tid < N) {
      // printf("index: %d %d %lf %lf\n", a_index, b_index, A[a_index], B[b_index]);
      // Each thread adds a single element
      B[tid] = A[index];
    }
}

namespace cuda 
{
ndarray::Array cudaSlices(const ndarray::Array &A, const ndarray::Slices &slices){
    ndarray::Slices filled_slices = ndarray::slices::filledSlices(A.shape(), slices);

    ndarray::Shape out_shape = ndarray::slices::getSliceOutShape(filled_slices);

    ndarray::LL start_index = ndarray::slices::getSliceStartIndex(filled_slices, A.stride());

    ndarray::Shape cum_mul_shape = ndarray::arrayutil::getCumulativeMultiShape(out_shape);

    ndarray::LL num_of_element = ndarray::arrayutil::getNumOfElementByShape(out_shape);



    double *B;
    cudaMalloc(&B, num_of_element*sizeof(double));

    int *stride;
    cudaMalloc(&stride, A.stride().size()*sizeof(int));
    cudaMemcpy(stride, &A.stride()[0], A.stride().size()*sizeof(int), cudaMemcpyHostToDevice);
    
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
    slice_<<<NUM_BLOCKS,NUM_THREADS>>>(A.deviceData(),
                                    B,
                                    num_of_element,
                                    stride,
                                    cum_shape,
                                    cum_mul_shape.size(),
                                    start_index
                                      );
    
    cudaFree(stride);
    cudaFree(cum_shape);
    ndarray::Shape filled_shape ;
    for(size_t i = 0; i < out_shape.size(); i++)
    {
      if(i< slices.size() && slices[i].size()<2) continue;
      filled_shape.push_back(out_shape[i]);
    }
    if(filled_shape.empty())filled_shape.push_back(ndarray::arrayutil::getNumOfElementByShape(out_shape));

    return ndarray::Array(filled_shape, nullptr, B);
}

}// end namespace cuda