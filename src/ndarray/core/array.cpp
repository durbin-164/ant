#include "array.h"
#include "cudaWrapper.h"
#include "cuda.h"
#include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
#include <stdlib.h>
#include <iostream>

namespace ndarray{

    Array::Array(const Shape &shape,  double *hdata, double *ddata,const bool onCuda)
    : shape_(shape), onCuda_(onCuda)
    {
        computeSize();
        updateStrides();

        if(hdata){
            hostData_ = hdata;
            isHostData_ = true;

            if(onCuda){
                isDeviceData_ = true;
                cudaMalloc((void **)&deviceData_, byteSize_);
                cudaMemcpy(deviceData_, hostData_, byteSize_, cudaMemcpyHostToDevice);
            }
        }else if(ddata){
            deviceData_ = ddata;
            isDeviceData_ = true;
        }
    }


    void Array::updateStrides(){
        stride_.resize(shape_.size());

        int initStride =1;

        for(LL i = shape_.size()-1 ; i>=0; i--){
            stride_[i] = initStride;
            initStride *= shape_[i];
        }
    }

    void Array::computeSize(){
        if(shape_.size()==0){
            size_ =0;
            return;
        }
        size_ =1;
        for(int s: shape_) size_ *= s;

        byteSize_ = size_ * sizeof(double);
    }

    Array::~Array()
    {
        if(isMallocHostData_){
            // TODO: complete array distructor to free host data.
            free(hostData_);
            hostData_=nullptr;
            isHostData_ = false;
        }
        if(isDeviceData_){
            unMappedToCuda();
        }
       
    }

    void Array::unMappedToCuda(){
        cudaFree(deviceData_);
        isDeviceData_= false;
        onCuda_= false;
    }

    void Array::mapDeviceDataToHost(){
        isMallocHostData_ = true;
        hostData_ = (double *)malloc(byteSize_);
        cudaMemcpy(hostData_, deviceData_, byteSize_, cudaMemcpyDeviceToHost);
    }


    // operator overload
    Array Array::operator+ (const Array &other)const {
        return cuda::cudaAdd(*this, other);
    }


    //attribute
     double* Array::hostData(){
        if(hostData_ == nullptr) 
            mapDeviceDataToHost();

        return hostData_;

    }

    void Array::setHostData(double *hdata){
        // TODO: need to set device data also
        if(hdata){
            hostData_ = hdata;
            isHostData_ = true;
        }
    }

    void Array::setDeviceData(double *ddata){
        // TODO need to set host data.
        if(ddata){
            deviceData_ = ddata;
            isDeviceData_ = true;
        }
     }


    void Array::setShape(Shape shape){
        // TODO: update this function
        shape_ = shape;
        updateStrides();
        computeSize();
      }


    
}