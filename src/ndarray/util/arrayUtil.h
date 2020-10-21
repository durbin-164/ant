#pragma once
#include "ndarray/core/dataType.h"
#include "ndarray/core/array.h"

namespace ndarray::arrayutil
{
/**
 * \brief get array size or number of element by array shape.<br/>
 * Example: shape{2,3,4} => output 2*3*4 = 24
 * @param shape vector data of array shape.
 * @return number of element of given array.
 */
ndarray::LL getNumOfElementByShape(const ndarray::Shape &shape);

/**
 * \brief get cumulative multyping shape. output size should be same as shape_size.<br/>
 * Example 1: shape{2,3,4} => cum_shape{12,4,1} <br/>
 * Example 2: shape{2,3,4,5}, offset = 2 => cun_shape{3,1}<br/>
 * Example 3: Shape{2,3},  offset=1 => cum_shape{1}
 * @param shape vector data of array shape.
 * @param offset skip number of element from last of the shape.
 * @return vector data with cumulative multyping shape likse as stride.
 */
ndarray::Shape getCumulativeMultiShape(const ndarray::Shape &shape, const int offset = 0 );


/**
 * \brief get matmul expected output shape from two array shape.
 * Also support broad castring.
 * Example: l_shape{4,1,3,4}, r_shape{1,3,4,5} => out_shape{4,3,3,5}
 * 
 * @param l_shape left array shape.
 * @param r_shape right array shape.
 * @return Expected broad casted matmul shape.
 */ 
ndarray::Shape getMatmulOutShape(const ndarray::Shape &l_shape, const ndarray::Shape &r_shape);

ndarray::LL getIndexFromIndices(const ndarray::Indices & indices, const ndarray::Array &A);
}