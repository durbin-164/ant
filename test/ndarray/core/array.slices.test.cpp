#include "gtest/gtest.h"
#include "array.h"
#include <vector>
#include "testUtil.h"

TEST(Slices, basicSlices)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{1,3,1}, {0,2,1}}];

    double *actual = B.hostData();
    double expected[] = {7,8,9,10, 13,14,15,16};
    VectorEQ(B.shape(), {2,2,2});
    DoubleArrayEQ(actual, expected, 8); 
}