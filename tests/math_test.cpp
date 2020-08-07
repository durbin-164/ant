#include "gtest/gtest.h"

#include "math.hpp"

// struct MathTest : public ::testing::Test {
//   virtual void SetUp() override {

//   }

//   virtual void TearDown() override {}
// };

TEST(MathTest, AddTwoNumber)
{
    // auto x = add(10,20);
    EXPECT_EQ(add(1, 1) ,2);
    EXPECT_EQ(subtract(1, 1) , 0);
}

TEST(MathTest, AddTheeNumber)
{
    // auto x = add(10,20);
    EXPECT_EQ(add(1, 1) ,2);
    EXPECT_EQ(subtract(1, 1) , 0);
}

// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }