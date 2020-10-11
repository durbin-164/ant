#include "gtest/gtest.h"
#include <vector>
#include "ndarray/core/dataType.h"
#include "ndarray/util/arrayUtil.h"
#include "testUtil.h"

TEST(getNumOfElementByShape, returnProperNumber)
{
    ndarray::Shape s = {2,3,4};

    ndarray::LL size = ndarray::getNumOfElementByShape(s);
    EXPECT_EQ(size, 24);
}


TEST(getCumulativeMultiShape, whenCumShapeZero)
{
    ndarray::Shape s = {4,2,3};

    ndarray::Shape actual = ndarray::getCumulativeMultiShape(s, 4);
    VectorEQ(actual, {4});

    s = {1};
    actual = ndarray::getCumulativeMultiShape(s);
    VectorEQ(actual, {});
    
}

TEST(getCumulativeMultiShape, returnProperCumShape)
{
    ndarray::Shape s = {5,4,2,3};

    ndarray::Shape actual = ndarray::getCumulativeMultiShape(s, 0);
    VectorEQ(actual, {24,6,3});
}