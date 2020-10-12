#include "gtest/gtest.h"
#include <vector>
#include <string>
#include <sstream>
#include "ndarray/exception/ndexception.h"

TEST(InvalidShapeException, returnProperMessage)
{
    ndarray::exception::InvalidSizeException IS("demo message");
    EXPECT_EQ(IS.what(),"demo message");

    ndarray::exception::InvalidShapeException IS2("demo", "(1,2,3)");
    std::string message = "demo operation could not possible with shape (1,2,3).";

    EXPECT_EQ(IS2.what(),message);

}

TEST(InvalidSizeException, returnProperMessage)
{
    ndarray::exception::InvalidSizeException IS("demo message");
    EXPECT_EQ(IS.what(),"demo message");
}