#include "gtest/gtest.h"
#include "array.h"
#include <vector>
#include "testUtil.h"
#include "ndarray/exception/ndexception.h"
#include <sstream>

TEST(Transpose, 2Dtranspose)
{
    ndarray::Shape a_shape = {5,4};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    ndarray::Array A(a_shape, a_data);

    A.transpose({1,0});
    double expected[][5] = {{1,5,9,13,17},{2,6,10,14,18},{3,7,11,15,19},{4,8,12,16,20}};
    VectorEQ(A.shape(), {4,5});
    VectorEQ(A.stride(), {1,4});
    
    for(int i=0;i<4; i++){
        for(int j=0; j< 5; j++){
            EXPECT_EQ(A(i,j), expected[i][j]);
        }
    }
}

TEST(Transpose, 3Dtranspose)
{
    ndarray::Shape a_shape = {3,4,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    ndarray::Array A(a_shape, a_data);

    A.transpose({2,0,1});
    double expected[][3][4] = { {{1,3,5,7},{9,11,13,15},{17,19,21,23}},
                                {{2,4,6,8},{10,12,14,16}, {18,20,22,24}}};
    
    VectorEQ(A.shape(), {2,3,4});
    VectorEQ(A.stride(), {1,8,2});
    
    for(int i=0;i<2; i++){
        for(int j=0; j< 3; j++){
            for(int k=0; k<4; k++){
                EXPECT_EQ(A(i,j,k), expected[i][j][k]);
            }
        }
    }
}

TEST(Transpose, exceptionWhenAxisNegative)
{
    ndarray::Shape a_shape = {3,4,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    ndarray::Array A(a_shape, a_data);

    EXPECT_THROW({
        try{
            A.transpose({2,-1,1});
        }catch(ndarray::exception::AxisOutOfRangeException &e){
            std::stringstream ss;
            ss<<"in transpose axis "<<-1<<" is out of range.";
            ss<<"axis must be between 0 and "<<2<<".";
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::AxisOutOfRangeException);

}

TEST(Transpose, exceptionWhenAxisMoreThenShapeSize)
{
    ndarray::Shape a_shape = {3,4,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    ndarray::Array A(a_shape, a_data);

    EXPECT_THROW({
        try{
            A.transpose({2,0,3});
        }catch(ndarray::exception::AxisOutOfRangeException &e){
            std::stringstream ss;
            ss<<"in transpose axis "<<3<<" is out of range.";
            ss<<"axis must be between 0 and "<<2<<".";
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::AxisOutOfRangeException);

}

TEST(Transpose, T_Add)
{
    ndarray::Shape a_shape = {3,2};
    ndarray::Shape b_shape = {3,2};

    double a_data[] ={1,2,3,4,5,6};
    double b_data[] = {13,15,9,8,3,10};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    A.transpose({1,0});
    B.transpose({1,0});

    ndarray::Array C = A+B;

    double *actual = C.hostData();
    double expected[] = {14,12,8,17,12,16};

    VectorEQ(C.shape(), {2,3});
    DoubleArrayEQ(actual, expected, 6); 
}

TEST(Transpose, T_Add_Broadcast)
{
    ndarray::Shape a_shape = {1,4};
    ndarray::Shape b_shape = {3,1};

    double a_data[] ={1,2,3,4};
    double b_data[] = {1,2,3};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    A.transpose({1,0});
    B.transpose({1,0});

    ndarray::Array C = A+B;

    double *actual = C.hostData();
    double expected[] = {2,3,4,3,4,5,4,5,6,5,6,7};

    VectorEQ(C.shape(), {4,3});
    DoubleArrayEQ(actual, expected, 12); 
}


TEST(Transpose, T_Matmul_BroadCast)
{
    ndarray::Shape a_shape = {2,3,1,2};
    ndarray::Shape b_shape = {3,2,2,1};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    double b_data[] ={12,11,10,9,8,7,6,5,4,3,2,1};
    ndarray::Array A(a_shape, a_data);
    ndarray::Array B(b_shape, b_data);

    A.transpose({1,2,3,0});
    B.transpose({3,2,1,0});

    auto C = A.matmul(B);

    double *actual = C.hostData();
    double expected[] = {82,50,18,104,64,24, 74,42,10,94,54,14,
                        126,78,30,148,92,36, 114,66,18,134,78,22,
                        170,106,42,192,120,48, 154,90,26,174,102,30};
    
    VectorEQ(C.shape(), {3,2,2,3});
    DoubleArrayEQ(actual, expected, 36); 
}

TEST(Transpose, T_slice)
{
    ndarray::Shape a_shape = {3,3,2};

    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    ndarray::Array A(a_shape, a_data);

    A.transpose({2,0,1});

    ndarray::Array B = A[{{1,2,1}, {0,2,1}}];

    double *actual = B.hostData();
    double expected[][2][3] = {{{2,4,6}, {8,10,12}}};
    VectorEQ(B.shape(), {1,2,3});
    
    for(int i=0;i<1; i++){
        for(int j=0; j< 2; j++){
            for(int k=0; k<3; k++){
                EXPECT_EQ(B(i,j,k), expected[i][j][k]);
            }
        }
    }
}