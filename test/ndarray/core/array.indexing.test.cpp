#include "gtest/gtest.h"
#include "array.h"
#include "ndarray/core/dataType.h"

TEST(Operator_Index, returnProperValue)
{
    ndarray::Shape shape= {2,3};
    double data[][3] = {{1,2,3},{10,20,30}};
    ndarray::Array A(shape, *data);
    double actual = A.getVal(0,1);
    EXPECT_EQ(actual, 1);
}