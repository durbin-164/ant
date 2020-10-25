#include "gtest/gtest.h"
#include "array.h"
#include <vector>
#include "testUtil.h"
#include "constants.h"

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

TEST(Slices, clipingSlice)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{1,5,1}, {0,-1},{1}}];

    double *actual = B.hostData();
    double expected[] = {8,10,14,16};
    VectorEQ(B.shape(), {2,2});
    DoubleArrayEQ(actual, expected, 4); 
}

TEST(Slices, singleValue)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{-2}, {2},{0}}];

    double *actual = B.hostData();
    double expected[] = {11};
    VectorEQ(B.shape(),{1});
    DoubleArrayEQ(actual, expected, 1); 
}

TEST(Slices, withStep)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{0,-1,2}, {-1,0,-2},{0,5,2}}];

    double *actual = B.hostData();
    double expected[] = {5};
    VectorEQ(B.shape(), {1,1,1});
    DoubleArrayEQ(actual, expected, 1); 
}

TEST(Slices, NoneTest)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{none, none}, {none, none ,none},{none, none}}];

    double *actual = B.hostData();
    double expected[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    VectorEQ(B.shape(), {3,3,2});
    DoubleArrayEQ(actual, expected, 18);
}

TEST(Slices, NoneReverse)
{
    ndarray::Shape a_shape = {5};

    double a_data[] = {1,2,3,4,5};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{none, none,-1}}];

    double *actual = B.hostData();
    double expected[] = {5,4,3,2,1};
    VectorEQ(B.shape(), {5});
    DoubleArrayEQ(actual, expected, 5);
}


TEST(Slices, NoneMixedTest)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    ndarray::Array B = A[{{2, none}, {none, -2 ,-1},{none, none}}];

    double *actual = B.hostData();
    double expected[] = {17,18};
    VectorEQ(B.shape(), {1,1,2});
    DoubleArrayEQ(actual, expected, 2);
}