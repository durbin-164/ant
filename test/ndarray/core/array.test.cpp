#include "gtest/gtest.h"
#include "array.h"
#include <vector>

void TestShapeData(Shape& actual, Shape &expected);

TEST(ArrayTest, ArrayInit)
{
    Shape s = {3};
    double a[] = {1,2,3};

    ndarray::Array A(s, a);

    EXPECT_EQ(A.rank(),1);
}

TEST(ArrayTest, AddTest)
{
    Shape s = {3};
    double a[] = {1,2,3};

    ndarray::Array A(s, a);
    ndarray::Array B(s, a);

    auto C = A + B;
    

    EXPECT_EQ(A.rank(),1);
    EXPECT_EQ(B.rank(), 1);
    EXPECT_EQ(C.rank(), 1);
    EXPECT_EQ(C.size(), 3);

    double * cActualData = C.hostData();
    double cExpectedData[] = {2,4,6};
    for(int i =0; i< C.size(); i++){
        EXPECT_EQ(cActualData[i], cExpectedData[i]);
    }
}


TEST(ArrayTest, AddTest2D)
{
    Shape s = {2,4};
    double a[][4] = {{1,2,3,4},
                    {5,6,7,8}};

    ndarray::Array A(s, *a);
    ndarray::Array B(s, *a);

    auto C = A + B;
    

    EXPECT_EQ(A.rank(),2);
    EXPECT_EQ(B.rank(), 2);
    EXPECT_EQ(C.rank(), 2);
    EXPECT_EQ(C.size(), 8);

    double * cActualData = C.hostData();
    double cExpectedData[] = {2,4,6,8,10,12,14,16};
    for(int i =0; i< C.size(); i++){
        EXPECT_EQ(cActualData[i], cExpectedData[i]);
    }
}


TEST(ArrayTest, TestAttributeFunction)
{
    Shape s = {2,4};
    double a[][4] = {{1,2,3,4},
                    {5,6,7,8}};

    ndarray::Array A(s, *a);
    
    EXPECT_EQ(A.rank(),2);
    EXPECT_TRUE(A.isHostData());
    EXPECT_TRUE(A.isDeviceData());

    auto shape = A.shape();
    TestShapeData(shape, s);

    auto stride = A.stride();
    Shape ex_stride = {4,1};
    std::cout<<stride.size()<<std::endl;
    TestShapeData(stride, ex_stride);

    Shape n_shape = {8};
    A.setShape(n_shape);
    auto a_n_s = A.shape();
    TestShapeData(a_n_s, n_shape);

    double n_data[]= {1,2,3,4,5,6,7,8};
    A.setHostData(n_data);
    auto a_n_data = A.hostData();

    for(int i=0; i<A.size(); i++){
        EXPECT_EQ(a_n_data[i], n_data[i]);
    }

    A.setShape({});
    A.updateStrides();
    A.computeSize();
    EXPECT_EQ(A.size(), 0);

    
    // TODO: need separate device data test.
    A.setDeviceData(n_data);
    A.setShape(s);
    EXPECT_EQ(A.rank(), 2);
    EXPECT_EQ(A.size(), 8);
    EXPECT_TRUE(A.isDeviceData());

}

void TestShapeData(Shape& actual, Shape& expected){

    for(int i=0; i<actual.size(); i++){
        EXPECT_EQ(actual[i], expected[i]);
    }
}
