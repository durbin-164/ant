#include "gtest/gtest.h"
#include "array.h"
#include <vector>
#include "testUtil.h"

TEST(Matmul, returnProperly)
{
    ndarray::Shape a_shape = {2,4};
    ndarray::Shape b_shape = {4,2};

    double a_data[] = {1,2,3,4,5,6,7,8};
    double b_data[] ={8,7,6,5,4,3,2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A.matmul(B);

    double *actual = C.hostData();
    double expected[] = {40,30,120,94};
    VectorEQ(C.shape(), {2,2});
    DoubleArrayEQ(actual, expected, 4); 
}

