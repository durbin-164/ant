#pragma once
#include "ndarray/core/dataType.h"

namespace ndarray
{
ndarray::LL getNumOfElementByShape(const ndarray::Shape &shape);
ndarray::Shape getCumulativeMultiShape(const ndarray::Shape &shape, const int offset = 0 );
ndarray::Shape getMatmulOutShape(const ndarray::Shape &l_shape, const ndarray::Shape &r_shape);
}