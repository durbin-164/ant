#include "gtest/gtest.h"
#include "ndarray/core/array.h"
#include <vector>
#include "ndarray/core/broadCasted.h"
#include "ndarray/util/util.h"
#include "testUtil.h"

TEST(paddedVactor, padProperly){

    ndarray::Shape s = {2,3};
    int size = 4;

    ndarray::Shape actual = ndarray::broadcast::paddedVector(s, size);
    VectorEQ(actual,{0,0,2,3});

    s = {3};
    size =1;
    actual = ndarray::broadcast::paddedVector(s, size);
    VectorEQ(actual,{3});


    ndarray::Stride st = {3,4};
    size = 5;
    ndarray::Stride st_actual = ndarray::broadcast::paddedVector(st, size, 1);
    VectorEQ(st_actual, {1,1,1,3,4});
}

TEST(paddedVactor, ThrowException){

    ndarray::Shape s = {2,3};
    int size = 1;

    EXPECT_THROW({

        try{
            ndarray::broadcast::paddedVector(s, size);
        }catch(const std::runtime_error &e){
            std::stringstream ss;
            ss <<"invalid padded size where expected size "<<size;
            ss<<" is less then give data size "<<s.size()<<".";
            EXPECT_EQ(ss.str(), e.what());
            throw;
        }

    },std::runtime_error);

}


TEST(getBroadCastedProperty, returnProperProperty)
{
    ndarray::Shape out_shape = {2,2,2,3};
    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    double b_data[] = {1,2,3};
    ndarray::Array A({2,2,2,3}, a_data);
    ndarray::Array B({1,3}, b_data);

    ndarray::broadcast::BroadCastedProperty BP = ndarray::broadcast::getBroadCastedProperty(out_shape, A, B);

    VectorEQ(BP.a_shape, {2,2,2,3});
    VectorEQ(BP.b_shape, {1,1,1,3});
    VectorEQ(BP.a_stride, {12,6,3,1});
    VectorEQ(BP.b_stride, {0,0,0,1});

}

TEST(getBroadCastedProperty, HorizontalAndVerticalBroadCast)
{
    ndarray::Shape out_shape = {4,3};
    double a_data[] = {1,2,3};
    double b_data[] = {1,2,3,4};

    ndarray::Array A({1,3}, a_data);
    ndarray::Array B({4,1}, b_data);

    ndarray::broadcast::BroadCastedProperty BP = ndarray::broadcast::getBroadCastedProperty(out_shape, A, B);

    VectorEQ(BP.a_shape, {1,3});
    VectorEQ(BP.b_shape, {4,1});
    VectorEQ(BP.a_stride, {0,1});
    VectorEQ(BP.b_stride, {1,0});
}


TEST(getBroadCastedProperty, withOffsetValue)
{
    ndarray::Shape out_shape = {2,2,2,3};
    double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    double b_data[] = {1,2,3};
    ndarray::Array A({2,2,2,3}, a_data);
    ndarray::Array B({1,3}, b_data);

    ndarray::broadcast::BroadCastedProperty BP = ndarray::broadcast::getBroadCastedProperty(out_shape, A, B,2);

    VectorEQ(BP.a_shape, {2,2,2,3});
    VectorEQ(BP.b_shape, {1,1,1,3});
    VectorEQ(BP.a_stride, {12,6,3,1});
    VectorEQ(BP.b_stride, {0,0,3,1});

}
