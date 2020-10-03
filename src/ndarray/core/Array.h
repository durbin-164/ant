#pragma once
#include "DataType.h"

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


        //Attributes
        inline double* hostData() const;
        inline void setHostData();
        inline double* deviceData() const;
        inline void setHostData();
        inline Shape stride() const;
        inline void setStride(Shape stride);


    private:
        double *deviceData_;
        double *hostData_;
        Shape shape_;
        bool onCuda_;
        Shape stride_;
        size_t size_;
        size_t byteSize_;


};

}//end of namespace ndarray




