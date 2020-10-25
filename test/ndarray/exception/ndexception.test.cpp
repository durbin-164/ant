#include "gtest/gtest.h"
#include <vector>
#include <string>
#include <sstream>
#include "ndarray/exception/ndexception.h"

TEST(InvalidShapeException, returnProperMessage)
{
    try{
        ndarray::exception::InvalidShapeException IS("demo message");
    }catch(ndarray::exception::InvalidShapeException &e){
         EXPECT_EQ(e.what(),"demo message");
    }

    try{
        ndarray::exception::InvalidShapeException IS2("demo", "(1,2,3)");
    }catch(ndarray::exception::InvalidShapeException &e){
        std::string message = "demo operation could not possible with shape (1,2,3).";
        EXPECT_EQ(e.what(),message);
    }

}

TEST(InvalidSizeException, returnProperMessage)
{
    try{
        ndarray::exception::InvalidSizeException IS("demo message");
    }catch(ndarray::exception::InvalidSizeException &e){
        EXPECT_EQ(e.what(),"demo message");
    }
}

TEST(IndexOutOfRangeException, returnProperMessage)
{
    try{
        ndarray::exception::IndexOutOfRangeException IORE("index out of range");
    }catch(ndarray::exception::IndexOutOfRangeException &e){
        EXPECT_EQ(e.what(),"index out of range");
    }
}


TEST(AxisOutOfRangeException, returnProperMessage)
{
    try{
        ndarray::exception::AxisOutOfRangeException AORE("axis out of range");
    }catch(ndarray::exception::AxisOutOfRangeException &e){
        EXPECT_EQ(e.what(),"axis out of range");
    }
    
    
}