#include "gtest/gtest.h"
#include "array.h"
#include <vector>
#include "testUtil.h"

TEST(Add, leftBroadCast)
{
    ndarray::Shape a_shape = {2,4};
    ndarray::Shape b_shape = {4};

    double a_data[] = {1,2,3,4,5,6,7,8};
    double b_data[] ={1,2,3,4};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A+B;

    double *actual = C.hostData();
    double expected[] = {2,4,6,8,6,8,10,12};
    VectorEQ(C.shape(), a_shape);
    DoubleArrayEQ(actual, expected, 8); 
}

TEST(Add, rightBroadCast)
{
    ndarray::Shape a_shape = {3};
    ndarray::Shape b_shape = {3,3};

    double a_data[] ={1,2,3};
    double b_data[] = {1,2,3,4,5,6,7,8,9};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A+B;

    double *actual = C.hostData();
    double expected[] = {2,4,6,5,7,9,8,10,12};

    VectorEQ(C.shape(), b_shape);
    DoubleArrayEQ(actual, expected, 9); 
}

TEST(Add, bothBroadCast)
{
    ndarray::Shape a_shape = {4,1};
    ndarray::Shape b_shape = {1,3};

    double a_data[] ={1,2,3,4};
    double b_data[] = {1,2,3};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A+B;

    double *actual = C.hostData();
    double expected[] = {2,3,4,3,4,5,4,5,6,5,6,7};

    VectorEQ(C.shape(), {4,3});
    DoubleArrayEQ(actual, expected, 12); 
}

TEST(Add, scalerArray)
{
    ndarray::Shape a_shape = {2,3};
    ndarray::Shape b_shape = {1};

    
    double a_data[] = {1,2,3,4,5,6};
    double b_data[] ={100};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A+B;

    double *actual = C.hostData();
    double expected[] = {101,102,103,104,105,106};

    VectorEQ(C.shape(), {2,3});
    DoubleArrayEQ(actual, expected, 6); 
}

TEST(Add, 1DAdd)
{
    ndarray::Shape a_shape = {5};
    ndarray::Shape b_shape = {5};

    
    double a_data[] = {1,2,3,4,5};
    double b_data[] ={10,20,30,40,50};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A+B;

    double *actual = C.hostData();
    double expected[] = {11,22,33,44,55};

    VectorEQ(C.shape(), {5});
    DoubleArrayEQ(actual, expected, 5); 
}

TEST(Add, 4DAdd)
{
    ndarray::Shape a_shape = {2,3,2,2};
    ndarray::Shape b_shape = {2,3,2,2};

    
    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    double b_data[] = {24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A+B;

    double *actual = C.hostData();
    double expected[] = {25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25};

    VectorEQ(C.shape(), {2,3,2,2});
    DoubleArrayEQ(actual, expected, 24); 
}