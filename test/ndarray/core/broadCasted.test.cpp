#include "gtest/gtest.h"
#include "array.h"
#include <vector>
#include "broadCasted.h"
#include "util.h"

void shapeTest(const ndarray::Shape &actual, const ndarray::Shape &expected){
    for(size_t i = 0; i<expected.size(); i++){
        EXPECT_EQ(actual[i], expected[i]);
    }
}

TEST(getBroadCastedShape, LeftShapeBroadCast)
{
    ndarray::Shape l_shape;
    ndarray::Shape r_shape;
    ndarray::Shape out_shape;
    l_shape = {1};
    r_shape = {3,4,2};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape);
    shapeTest(out_shape, r_shape);

    l_shape = {4,2};
    r_shape = {3,4,2};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape);
    shapeTest(out_shape, r_shape);

    l_shape = {1,2};
    r_shape = {3,4,2};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape,2);
    shapeTest(out_shape, {3});
  
}

TEST(getBroadCastedShape, RightShapeBroadCast)
{
    ndarray::Shape l_shape;
    ndarray::Shape r_shape;
    ndarray::Shape out_shape;
    l_shape = {4,5,2};
    r_shape = {1};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape);
    shapeTest(out_shape, l_shape);

    l_shape = {4,5,3};
    r_shape = {3};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape);
    shapeTest(out_shape,l_shape);

    l_shape = {4,5,3};
    r_shape = {2};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape,1);
    shapeTest(out_shape, {4,5});

    l_shape = {4,5,3};
    r_shape = {4,5,3};
    out_shape = ndarray::getBroadCastedShape(l_shape, r_shape,0);
    shapeTest(out_shape, l_shape);
}

TEST(getBroadCastedShape, EmptyShapeException)
{
    ndarray::Shape l_shape;
    ndarray::Shape r_shape;
    ndarray::Shape out_shape;
    std::stringstream ss;
    l_shape = {};
    r_shape = {1,2,3};

    EXPECT_THROW({
                try{
                    ndarray::getBroadCastedShape(l_shape, r_shape);
                }catch(std::runtime_error& e){
                    ss<<"operands could not be broadcast together with shapes(";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, std::runtime_error);


    l_shape = {2,3};
    r_shape = {};
    ss.str(std::string());

    EXPECT_THROW({
                try{
                    ndarray::getBroadCastedShape(l_shape, r_shape);
                }catch(std::runtime_error& e){
                    ss<<"operands could not be broadcast together with shapes(";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, std::runtime_error);      

    
    l_shape = {};
    r_shape = {};
    ss.str(std::string());

    EXPECT_THROW({
                try{
                    ndarray::getBroadCastedShape(l_shape, r_shape);
                }catch(std::runtime_error& e){
                    ss<<"operands could not be broadcast together with shapes(";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, std::runtime_error);
}


TEST(getBroadCastedShape, UnableBroadCastedExpection)
{
    ndarray::Shape l_shape;
    ndarray::Shape r_shape;
    ndarray::Shape out_shape;
    l_shape = {3,2};
    r_shape = {1,2,3};

    EXPECT_THROW({
                try{
                    ndarray::getBroadCastedShape(l_shape, r_shape);
                }catch(std::runtime_error& e){
                    std::stringstream ss;
                    ss<<"operands could not be broadcast together with shapes(";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    throw;
                }
            }, std::runtime_error);

}

