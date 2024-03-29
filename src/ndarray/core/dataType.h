#pragma once
#include<vector>

namespace ndarray
{

/*!
 * Integer Type vector data structure.
 */
using Shape = std::vector<int>;
/*!
 * Integer Type vector data structure.
 */
using Stride = std::vector<int>;
/*!
 * Integer Type vector data structure.
 */
using Axis  = std::vector<int>;

/*!
 * Integer Type vector data structure.
 */
using Indices  = std::vector<int>;

/*!
 * Long Long data structure.
 */
using LL = long long;

/*!
 * Integer Type 2D-vector data structure.
 */
using Slices  = std::vector<std::vector<int>>;


}//end ndarray namespace
