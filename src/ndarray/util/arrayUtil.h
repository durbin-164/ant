#pragma once
#include "ndarray/core/dataType.h"

namespace ndarray
{
ndarray::LL getNumOfElementByShape(const ndarray::Shape &shape);
ndarray::Shape getCumulativeMultiShape(const ndarray::Shape &shape, const int offset = 0 );
}