#include "ndarray/cuda/util.cuh"

__device__ int getIndex(int index, int *cum_mul_shape, int *stride , int cum_shape_N){
    int ret_index =0;
    
    for(int i=0; i<cum_shape_N; i++){
        ret_index += (index/cum_mul_shape[i])*stride[i];
        index %= cum_mul_shape[i];
    }
    ret_index += index * stride[cum_shape_N];

    return ret_index;
}


