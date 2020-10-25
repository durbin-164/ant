#pragma once
#include "gtest/gtest.h"
#include <vector>

inline void VectorEQ(const std::vector<int> &actual, const std::vector<int>& expected){
    EXPECT_EQ(actual.size(), expected.size());

    for(size_t i =0; i< actual.size(); i++){
        EXPECT_EQ(actual[i], expected[i])<<"where i = "<<i;
    }
}

inline void VectorEQ2D(const std::vector<std::vector<int>> &actual, const std::vector<std::vector<int>> & expected){
    EXPECT_EQ(actual.size(), expected.size());

    for(size_t i =0; i< actual.size(); i++){
        for(size_t j =0; j<actual[i].size(); j++){
             EXPECT_EQ(actual[i][j], expected[i][j])<<"where i= "<<i<<" and j = "<<j;
        }
    }
}

inline void DoubleArrayEQ(const double *actual, const double *expected, const int N){
    for(size_t i =0; i<N; i++){
        EXPECT_EQ(actual[i], expected[i])<<"where i = "<<i;
    }
}