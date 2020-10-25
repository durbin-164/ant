#include "gtest/gtest.h"
#include "ndarray/core/array.h"
#include <vector>
#include "testUtil.h"
#include "ndarray/exception/ndexception.h"
#include "ndarray/core/slices.h"

TEST(filledSlices, handelCase1NagitiveValue)
{
    ndarray::Shape in_shape = {3,4,2};
    ndarray::Slices slices = {{-2}, {-1}, {0}};

    ndarray::Slices actual = ndarray::slices::filledSlices(in_shape, slices);
   
    ndarray::Slices expected = {{1,2,1}, {3,4,1}, {0,1,1}};
    VectorEQ2D(actual, expected);
}

TEST(filledSlices, handelCase2NagitiveValue)
{
    ndarray::Shape in_shape = {3,4,2};
    ndarray::Slices slices = {{-4,4}, {1,-5,2}, {-2,4,2}};

    ndarray::Slices actual = ndarray::slices::filledSlices(in_shape, slices);
    
    ndarray::Slices expected = {{0,3,1}, {1,0,2}, {0,2,2}};
    VectorEQ2D(actual, expected);
}

TEST(filledSlices, expectionWhenSizeMissMatch)
{
    ndarray::Shape in_shape = {3,4};
    ndarray::Slices slices = {{-4,4}, {1,-5,2}, {-2,4,2}};
    EXPECT_THROW({
        try{
            ndarray::Slices actual = ndarray::slices::filledSlices(in_shape, slices);
        }catch(ndarray::exception::InvalidSizeException &e){
            std::stringstream ss;
            ss <<"slices size "<<slices.size();
            ss<<" which is greater than input shape size "<<in_shape.size()<<".";
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::InvalidSizeException);
    
}

TEST(filledSlices, expectionWhenoutOfRangeIndex)
{
    ndarray::Shape in_shape = {3,4,2};
    ndarray::Slices slices = {{-4,4}, {1,-5,2}, {-4}};
    EXPECT_THROW({
        try{
            ndarray::Slices actual = ndarray::slices::filledSlices(in_shape, slices);
        }catch(ndarray::exception::IndexOutOfRangeException &e){
            std::stringstream ss;
            ss<<"index "<<slices[2][0]<<" is out of bounds for axis "<<2<<" with size "<<in_shape[2];
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::IndexOutOfRangeException);

    in_shape = {3,4,2};
    slices = {{-4,4}, {4}, {1}};
    EXPECT_THROW({
        try{
            ndarray::Slices actual = ndarray::slices::filledSlices(in_shape, slices);
        }catch(ndarray::exception::IndexOutOfRangeException &e){
            std::stringstream ss;
            ss<<"index "<<slices[1][0]<<" is out of bounds for axis 1 with size "<<in_shape[1];
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::IndexOutOfRangeException);
    
}


TEST(filledSlices, expectionParameterSize)
{
    ndarray::Shape in_shape = {3,4,2};
    ndarray::Slices slices = {{-2}, {-1,4,3,4}, {0}};
    EXPECT_THROW({
        try{
            ndarray::Slices actual = ndarray::slices::filledSlices(in_shape, slices);
        }catch(ndarray::exception::InvalidSizeException &e){
            std::stringstream ss;
            ss<<"slice parameter size "<<4<<" out of range at axis 1";
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },ndarray::exception::InvalidSizeException);
    
}

TEST(getSliceOutShape, returnProperOutput)
{
    ndarray::Slices in_slices = {{2,-4,2},{0,3,2}, {4,0,-1}};

    ndarray::Shape actual = ndarray::slices::getSliceOutShape(in_slices);
    
    ndarray::Shape expected = {0,2,4};
    VectorEQ(actual,expected);
}

TEST(getSliceStartIndex, returnProperValue)
{
    ndarray::Slices in_slices = {{2,-4,2},{0,3,2}, {4,0,-1}};
    ndarray::Stride in_stride = {6,2,1};

    ndarray::LL actual = ndarray::slices::getSliceStartIndex(in_slices,in_stride);

    EXPECT_EQ(actual,16);

}