#include "Array.h"
#include "cuda.h"
#include <cuda_runtime_api.h>

namespace ndarray{

    Array::Array(const Shape &shape,  double *hdata, double *ddata,const bool onCuda)
    {
        shape_ = shape;
        onCuda_ = onCuda;
        computeSize();
        updateStrides();

        if(hdata){
            hostData_ = hdata;

            if(onCuda){
                cudaMalloc((void **)&deviceData_, byteSize_);
                cudaMemcpy(deviceData_, hostData_, byteSize_, cudaMemcpyHostToDevice);
            }
        }else if(ddata){
            deviceData_ = ddata;
        }
    }


    void Array::updateStrides(){
        stride_.resize(shape_.size());

        LL initStride =1;

        for(LL i = shape_.size()-1 ; i>=0; i--){
            stride_.push_back(initStride);
            initStride *= shape_[i];
        }
    }

    void Array::computeSize(){
        if(shape_.size()==0){
            size_ =0;
            return;
        }
        size_ =1;
        for(LL i = 0; i < shape_.size(); i++) size_ *= shape_[i];

        byteSize_ = size_ * sizeof(double);
    }

    
}