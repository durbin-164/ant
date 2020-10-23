#pragma onece

#include "ndarray/core/dataType.h"

namespace ndarray::slices
{
ndarray::Slices filledSlices(const ndarray::Shape &in_shape, const ndarray::Slices &slices);

ndarray::Shape getSliceOutShape(const ndarray::Slices &slices);

ndarray::LL getSliceStartIndex(const ndarray::Slices &slices, const ndarray::Stride &stride);
}