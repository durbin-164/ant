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

TEST(Matmul, leftBroadCast)
{
    ndarray::Shape a_shape = {2,3,2,2};
    ndarray::Shape b_shape = {2,3};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    double b_data[] ={6,5,4,3,2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A.matmul(B);

    double *actual = C.hostData();
    double expected[] = {12,9 ,6,30,23,16, 48,37,26,66,51,36,
                        84,65,46,102,79,56, 120,93,66,138,107,76,
                        156,121,86,174,135,96, 192,149,106,210,163,116};
    
    VectorEQ(C.shape(), {2,3,2,3});
    DoubleArrayEQ(actual, expected, 36); 
}

TEST(Matmul, rightBroadCast)
{
    ndarray::Shape a_shape = {2,2};
    ndarray::Shape b_shape = {2,2,3};

    double a_data[] = {1,2,3,4};
    double b_data[] ={12,11,10,9,8,7,6,5,4,3,2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A.matmul(B);

    double *actual = C.hostData();
    double expected[] = {30,27,24,72,65,58, 12,9,6,30,23,16};
    
    VectorEQ(C.shape(), {2,2,3});
    DoubleArrayEQ(actual, expected, 12); 
}


TEST(Matmul, 2Dto1D)
{
    ndarray::Shape a_shape = {1,2};
    ndarray::Shape b_shape = {2,1};

    double a_data[] = {1,2};
    double b_data[] ={2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A.matmul(B);

    double *actual = C.hostData();
    double expected[] = {4};
    
    VectorEQ(C.shape(), {1,1});
    DoubleArrayEQ(actual, expected, 1); 
}


TEST(Matmul, bothBroadCast)
{
    ndarray::Shape a_shape = {3,1,2,2};
    ndarray::Shape b_shape = {1,2,2,3};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    double b_data[] ={12,11,10,9,8,7,6,5,4,3,2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    auto C = A.matmul(B);

    double *actual = C.hostData();
    double expected[] = {30,27,24,72,65,58, 12,9,6,30,23,16,
                        114,103,92,156,141,126, 48,37,26,66,51,36,
                        198,179,160,240,217,194, 84,65,46,102,79,56};
    
    VectorEQ(C.shape(), {3,2,2,3});
    DoubleArrayEQ(actual, expected, 36); 
}