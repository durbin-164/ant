#pragma once
#include "dataType.h"
#include "array.h"

namespace ndarray
{

struct BroadCastedProperty;
Shape getBroadCastedShape(const Shape &l_shape, const Shape &r_shape, const int offset = 0);

BroadCastedProperty getBroadCastedProperty(const ndarray::Shape &out_shape, const ndarray::Array &A, const ndarray::Array &B,const int offset = 0);
std::vector<int>paddedVector(const std::vector<int>&vec, const int size, const int pad_value = 0);

struct BroadCastedProperty{
    Shape a_shape;
    Shape b_shape;
    Stride a_stride;
    Stride b_stride;

    BroadCastedProperty(Shape &a_shape_, Shape &b_shape_, Stride &a_stride_, Stride &b_stride_ ){
        a_shape = a_shape_;
        b_shape = b_shape_;
        a_stride = a_stride_;
        b_stride = b_stride_;
    }
};

}// need ndarray name space