#include "gtest/gtest.h"
#include "ndarray/core/array.h"
#include <vector>
#include "ndarray/core/broadCasted.h"
#include "ndarray/util/util.h"
#include "testUtil.h"
#include "ndarray/exception/ndexception.h"

TEST(getBroadCastedShape, LeftShapeBroadCast)
{
    ndarray::Shape l_shape;
    ndarray::Shape r_shape;
    ndarray::Shape out_shape;
    l_shape = {1};
    r_shape = {3,4,2};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
    VectorEQ(out_shape, r_shape);

    l_shape = {4,2};
    r_shape = {3,4,2};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
    VectorEQ(out_shape, r_shape);

    l_shape = {1,2};
    r_shape = {3,4,2};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape,2);
    VectorEQ(out_shape, {3});
  
}

TEST(getBroadCastedShape, RightShapeBroadCast)
{
    ndarray::Shape l_shape;
    ndarray::Shape r_shape;
    ndarray::Shape out_shape;
    l_shape = {4,5,2};
    r_shape = {1};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
    VectorEQ(out_shape, l_shape);

    l_shape = {4,5,3};
    r_shape = {3};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
    VectorEQ(out_shape,l_shape);

    l_shape = {4,5,3};
    r_shape = {2};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape,1);
    VectorEQ(out_shape, {4,5});

    l_shape = {4,5,3};
    r_shape = {4,5,3};
    out_shape = ndarray::broadcast::getBroadCastedShape(l_shape, r_shape,0);
    VectorEQ(out_shape, l_shape);
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
                    ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
                }catch(ndarray::exception::InvalidShapeException& e){
                    ss<<"broadcast operation could not possible with shape (";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, ndarray::exception::InvalidShapeException);


    l_shape = {2,3};
    r_shape = {};
    ss.str(std::string());

    EXPECT_THROW({
                try{
                    ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
                }catch(ndarray::exception::InvalidShapeException& e){
                    ss<<"broadcast operation could not possible with shape (";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, ndarray::exception::InvalidShapeException);    

    
    l_shape = {};
    r_shape = {};
    ss.str(std::string());

    EXPECT_THROW({
                try{
                    ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
                }catch(ndarray::exception::InvalidShapeException& e){
                    ss<<"broadcast operation could not possible with shape (";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, ndarray::exception::InvalidShapeException);


    
    l_shape = {0};
    r_shape = {1,2};
    ss.str(std::string());

    EXPECT_THROW({
                try{
                    ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
                }catch(ndarray::exception::InvalidShapeException& e){
                    ss<<"broadcast operation could not possible with shape (";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, ndarray::exception::InvalidShapeException);

    l_shape = {1,2,3};
    r_shape = {0};
    ss.str(std::string());

    EXPECT_THROW({
                try{
                    ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
                }catch(ndarray::exception::InvalidShapeException& e){
                    ss<<"broadcast operation could not possible with shape (";
                    ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, ndarray::exception::InvalidShapeException);
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
                    ndarray::broadcast::getBroadCastedShape(l_shape, r_shape);
                }catch(ndarray::exception::InvalidShapeException& e){
                    std::stringstream ss;
                    ss<<"broadcast operation could not possible with shape (";
                    ss<<ndarray::getVectorIntInString(r_shape)<<") (";
                    ss<<ndarray::getVectorIntInString(l_shape)<<").";
                    EXPECT_EQ(ss.str(), e.what() );
                    throw;
                }
            }, ndarray::exception::InvalidShapeException);

}

