#include "gtest/gtest.h"
#include <vector>
#include "dataType.h"
#include "util.h"

TEST(getVectorIntInString, CheckStringIsProuceCorrectly)
{
    ndarray::Shape s = {3,1};
    std::string actual = ndarray::getVectorIntInString(s);

    EXPECT_EQ("3,1", actual);

    s = {};
    actual = ndarray::getVectorIntInString(s);

    EXPECT_EQ("", actual);
    
}