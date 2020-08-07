#include "gtest/gtest.h"
#include <arrayfire.h>
#include "basic.hpp"


TEST(BasicTest, AddTwoNumber)
{
    auto a = af:: randu(1,3);
    auto b = af:: randu(1,3);
    auto c = af:: randu(1,3);

    ant:: basic* bsc = new ant::basic(a, b);

    auto d = bsc ->sum();
    // // auto x = add(10,20);
    EXPECT_EQ(d.dims(),c.dims());
    // af::ASSERT_ARRAYS_EQUAL(d,c);
    // EXPECT_EQ(subtract(1, 1) , 0);
}
