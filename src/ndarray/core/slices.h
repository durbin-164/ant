#pragma onece

#include "ndarray/core/dataType.h"

namespace ndarray::slices
{
/**
 * \brief return proper format 2D slices vector<br/>
 * Example: <br/>
 * in_shape{3,4,2}, slices{{0,2}, {1,-1,2}} => out_slices{{0,2,1},{1,3,2},{0,2,1}}<br/>
 * in_shape{3,4,2}, slices{{0,none}, {none,-1,2}, {none,none,-1}} => out_slices{{0,4,1},{0,3,2},{1,-1,-1}}<br/>
 * 
 * @param in_shape actual shape of the Slices ndarray.
 * @param slices 2D vector with slices value. all inner vector must be size less than or equal one.
 * @return 2D slices vector with proper value and size must be equal in_shape size.
 */
ndarray::Slices filledSlices(const ndarray::Shape &in_shape, const ndarray::Slices &slices);

/**
 * \brief return primary output shape of filtered slices<br/>
 * Example: slices{{0,2,1},{1,3,2},{0,2,1}} => out_shape{2,1,2}
 * 
 * @param slices 2D slices vector
 * @return shape of the output ndarray.
 */
ndarray::Shape getSliceOutShape(const ndarray::Slices &slices);

/**
 * \brief return slices base index. from where index will be start.<br/>
 * Example: slices{{2,5,1},{0,3,2}} , stride{6,1} => out = 6*2+0*1 = 12
 * 
 * @param slices 2D slices vector
 * @param stride actual stride of slices ndarray
 * @return start index where from slices will be start.
 */
ndarray::LL getSliceStartIndex(const ndarray::Slices &slices, const ndarray::Stride &stride);
}