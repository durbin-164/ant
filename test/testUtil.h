#pragma once
#include "gtest/gtest.h"
#include <vector>

inline void testVector(const std::vector<int> &actual, const std::vector<int>& expected){
    EXPECT_EQ(actual.size(), expected.size());

    for(size_t i =0; i< actual.size(); i++){
        EXPECT_EQ(actual[i], expected[i]);
    }
}