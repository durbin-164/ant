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

__global__ void matmul_(double* A,
    double* B,
    double* C,
    long long N,
    int *a_stride,
    int *b_stride,
    int stride_N,
    int *cum_shape,
    int cum_shape_N,
    int m,
    int k,
    int n,
    long long batch
    ) {

    for(long long b = 0; b<batch; b++){

        int row = blockIdx.y * blockDim.y + threadIdx.y; 
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        double sum = 0;
        if( col < n && row < m) 
        {
            for(int i = 0; i < k; i++) 
            {
                int a_index = getIndex(b, cum_shape, a_stride, cum_shape_N)+ a_stride[stride_N-2]*row+ a_stride[stride_N-1]*i;
                int b_index = getIndex(b, cum_shape, b_stride, cum_shape_N)+ b_stride[stride_N-2]*i+ b_stride[stride_N-1]*col;
    
                // printf("%d %d %d %d %d %lf %lf\n",b,row,col, a_index, b_index,A[a_index] , B[b_index] );
                sum += A[a_index] * B[b_index];
                  
            }
            if((b*m*n)+row * n + col< batch*m*n)
                C[(b*m*n)+row * n + col] = sum;
        }
    
    }
}

namespace cuda
{

ndarray::Array cudaMatmul(const ndarray::Array &A, const ndarray::Array &B){
    int offset = 2;
    ndarray::Shape out_shape = ndarray::arrayutil::getMatmulOutShape(A.shape(), B.shape());

    ndarray::broadcast::BroadCastedProperty BP = ndarray::broadcast::getBroadCastedProperty(out_shape, A, B, offset);

    ndarray::Shape cum_mul_shape = ndarray::arrayutil::getCumulativeMultiShape(out_shape, offset);

    ndarray:: LL num_of_element = ndarray::arrayutil::getNumOfElementByShape(out_shape);
    ndarray:: LL batch_size = ndarray::arrayutil::getNumOfElementByShape({out_shape.begin(), out_shape.end()-2});

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

    int m = out_shape.end()[-2];
    int n = out_shape.end()[-1];
    int k = A.shape().end()[-1];

    const int BLOCK_SIZE =16;

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmul_<<<dimGrid, dimBlock>>>(A.deviceData(),
                                        B.deviceData(),
                                        C,
                                        num_of_element,
                                        a_stride,
                                        b_stride,
                                        BP.a_stride.size(),
                                        cum_shape,
                                        cum_mul_shape.size(),
                                        m,
                                        k,
                                        n,
                                        batch_size
                                            );

    cudaFree(a_stride);
    cudaFree(b_stride);
    cudaFree(cum_shape);

    if(out_shape.begin()[0]==1 && (out_shape.size() > A.shape().size() && out_shape.size() >B.shape().size())) 
        out_shape = {out_shape.begin()+1, out_shape.end()};
    
    return ndarray::Array(out_shape, nullptr, C);
}

}//end cuda namespace