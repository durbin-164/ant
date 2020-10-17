#include "gtest/gtest.h"
#include "array.h"
#include "ndarray/core/dataType.h"

TEST(Operator_Index, returnProperValue)
{
    ndarray::Shape shape= {2,3};
    double data[][3] = {{1,2,3},{10,20,30}};
    ndarray::Array A(shape, *data);
    
    EXPECT_EQ(A(0,0), 1);
    EXPECT_EQ(A(0,1), 2);
    EXPECT_EQ(A(0,2), 3);

    EXPECT_EQ(A(1,0), 10);
    EXPECT_EQ(A(1,1), 20);
    EXPECT_EQ(A(1,2), 30);
}