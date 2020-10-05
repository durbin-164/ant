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
        Array operator+(Array &other);


        //Attributes
        double* hostData();
        inline void setHostData();
        inline double* deviceData() const {return deviceData_;}
        inline void setDeviceData();
        inline Shape stride() const {return stride_;}
        inline void setStride(Shape stride);
        inline int rank() const {return shape_.size();}
        inline bool isHostData() const {return isHostData_;}
        inline void setIsHostData(bool hostData);
        inline bool isDeviceData() const {return isDeviceData_;}
        inline Shape shape() const {return shape_;}
        inline int size() const {return size_;}


    private:
        double *deviceData_ = nullptr;
        double *hostData_ = nullptr;
        Shape shape_;
        bool onCuda_;
        Shape stride_;
        size_t size_;
        size_t byteSize_;
        bool isHostData_;
        bool isDeviceData_;
        bool isMallocHostData_= false;


};

}//end of namespace ndarray




