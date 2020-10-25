#include "gtest/gtest.h"
#include "array.h"
#include "ndarray/core/dataType.h"
#include "ndarray/exception/ndexception.h"
#include <sstream>

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


TEST(Operator_Index, exceptSizeMissMatch)
{
    ndarray::Shape shape= {2,3};
    double data[][3] = {{1,2,3},{10,20,30}};
    ndarray::Array A(shape, *data);
    
    EXPECT_THROW({
        try{
            A(0);
        }catch(ndarray::exception::InvalidSizeException &e){
            std::stringstream ss;
            ss<<"invalid index. index size must be 2.";
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::InvalidSizeException);
}

TEST(Operator_Index, indexOutOfRange)
{
    ndarray::Shape shape= {2,3};
    double data[][3] = {{1,2,3},{10,20,30}};
    ndarray::Array A(shape, *data);
    
    EXPECT_THROW({
        try{
            A(0,3);
        }catch(ndarray::exception::IndexOutOfRangeException &e){
            std::stringstream ss;
             ss<<"index 3 must be between 0 and 2 at axis 1.";
             EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::IndexOutOfRangeException);
}