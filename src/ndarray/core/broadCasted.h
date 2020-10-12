/**
 * @file broadCasted.h
 *
 * Interfaces to Data Buffer.
 *
 * This header define the interfaces of general purpose dynamic data buffer that
 * implemented by Equinox.
 */

#pragma once
#include "ndarray/core/dataType.h"
#include "ndarray/core/array.h"

namespace ndarray
{

struct BroadCastedProperty;

// \example: l_shape {4,1}, r_shape{1,3} => out_shape{4,3}


/////////////////////////////////////////////////////////////////////// 
/// \brief getBroadCastedShape
/// \param l_shape  Left array shape.
/// \param r_shape  Right array shape.
/// \param offset int value
/// \return Operation output shape.
/////////////////////////////////////////////////////////////////////////
ndarray::Shape getBroadCastedShape(const ndarray::Shape &l_shape,const ndarray::Shape &r_shape, const int offset = 0);


/*!
 * \brief Calculate broadcasted operated array shape and stride.
 *        Which is needed for calculate actual index of operation data.
 * @param out_shape  broadcasted output shape.
 * @param A  ndarray.
 * @param B  ndarray.
 * @return BroadCastedProperty DTO class which have a_shape, b_shape, a_stride, b_stride.
 */
BroadCastedProperty getBroadCastedProperty(const ndarray::Shape &out_shape, const ndarray::Array &A, const ndarray::Array &B,const int offset = 0);\

/*!
 * \brief make a vector fill with pad value
 * @param vec int type vector.
 * @param size expected vector size.
 * @param pad_value expected value to fill. default: 0
 * 
 * @return vector with padded value
 * 
 * Example: vec{3,2,4}, size = 5, pad_value = 100 =>out_vec{100,100,3,2,4}
 */
std::vector<int>paddedVector(const std::vector<int>&vec, const int size, const int pad_value = 0);

struct BroadCastedProperty{
    ndarray::Shape a_shape;
    ndarray::Shape b_shape;
    ndarray::Stride a_stride;
    ndarray::Stride b_stride;

    BroadCastedProperty(ndarray::Shape &a_shape_, ndarray::Shape &b_shape_, ndarray::Stride &a_stride_, ndarray::Stride &b_stride_ ){
        a_shape = a_shape_;
        b_shape = b_shape_;
        a_stride = a_stride_;
        b_stride = b_stride_;
    }
};

}// need ndarray name space