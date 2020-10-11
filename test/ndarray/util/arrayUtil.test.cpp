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

TEST(getMatmulOutShape, returnProperShape)
{
    ndarray::Shape l_shape = {2,3,4};
    ndarray::Shape r_shape = {2,4,9};

    ndarray::Shape actual = ndarray::getMatmulOutShape(l_shape,r_shape);

    VectorEQ(actual, {2,3,9});
}

TEST(getMatmulOutShape, leftBroadCast)
{
    ndarray::Shape l_shape = {3,4};
    ndarray::Shape r_shape = {2,4,9};

    ndarray::Shape actual = ndarray::getMatmulOutShape(l_shape,r_shape);

    VectorEQ(actual, {2,3,9});
}

TEST(getMatmulOutShape, rightBroadCast)
{
    ndarray::Shape l_shape = {3,3,4};
    ndarray::Shape r_shape = {4,9};

    ndarray::Shape actual = ndarray::getMatmulOutShape(l_shape,r_shape);

    VectorEQ(actual, {3,3,9});
}

TEST(getMatmulOutShape, bothBroadCast)
{
    ndarray::Shape l_shape = {3,1,3,4};
    ndarray::Shape r_shape = {1,4,4,9};

    ndarray::Shape actual = ndarray::getMatmulOutShape(l_shape,r_shape);

    VectorEQ(actual, {3,4,3,9});
}

TEST(getMatmulOutShape, givenBatch)
{
    ndarray::Shape l_shape = {2,4};
    ndarray::Shape r_shape = {4,2};

    ndarray::Shape actual = ndarray::getMatmulOutShape(l_shape,r_shape);

    VectorEQ(actual, {1,2,2});
}