#pragma once
#include "dataType.h"

namespace ndarray{
class Array
{
    public:
        Array();
        ~Array();

        /**
         * \brief Initialize an array
         * 
         * @param shape  vector of shape.
         * @param hdata  double pointer reference of host data. default nullprt
         * @param ddata  double pointer reference of device data. default nullprt
         * @param ongpu  boolen variable for on gpu or on cpu.
         */
        Array(const Shape &shape, double *hdata = nullptr, double *ddata = nullptr,const bool onCuda = true);


        //properties function
        void updateStrides();
        void computeSize();
        void unMappedToCuda();
        void mapDeviceDataToHost();


        //operator overload
        Array operator+(const Array &other) const;


        //Attributes
        double* hostData();
        void setHostData(double *hdata);
        double* deviceData() const {return deviceData_;}
        void setDeviceData(double *ddata);
        Stride stride() const {return stride_;}
        size_t rank() const {return shape_.size();}
        bool isHostData() const {return isHostData_;}
        void setIsHostData(bool isHostData){isHostData_ = isHostData;}
        bool isDeviceData() const {return isDeviceData_;}
        void setIsDeviceData(bool isDeviceData){isDeviceData_= isDeviceData;}
        Shape shape() const {return shape_;}
        void setShape(Shape shape);
        size_t size() const {return size_;}


    private:
        double *deviceData_ = nullptr;
        double *hostData_ = nullptr;
        Shape shape_;
        bool onCuda_;
        Stride stride_;
        size_t size_;
        size_t byteSize_;
        bool isHostData_;
        bool isDeviceData_;
        bool isMallocHostData_= false;


};

}//end of namespace ndarray




