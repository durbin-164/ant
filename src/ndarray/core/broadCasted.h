#pragma once
#include "dataType.h"
#include "array.h"

namespace ndarray
{

struct BroadCastedProperty;

Shape getBroadCastedShape(const Shape &l_shape, const Shape &r_shape, const int offset = 0);

BroadCastedProperty getBroadCastedProperty(Shape out_shape, const ndarray::Array &A, const ndarray::Array &B,const int offset = 0);

}// need ndarray name space