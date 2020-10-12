#include "ndarray/util/arrayUtil.h"
#include <stdexcept> 
#include <sstream>
#include "ndarray/util/util.h"
#include "ndarray/core/broadCasted.h"
#include <iostream>
#include "ndarray/exception/ndexception.h"

namespace ndarray
{
ndarray::LL ndarray::arrayutil::getNumOfElementByShape(const ndarray::Shape &shape){

    ndarray::LL size = 1;
    for(ndarray:: LL x : shape) size *= x;

    return size;
}


ndarray::Shape ndarray::arrayutil::getCumulativeMultiShape(const ndarray::Shape &shape, const int offset){
    ndarray::Shape cum_shape(std::max((int)shape.size()-offset-1,0));

    if(shape.size()-offset <=1)return cum_shape;

    if(cum_shape.size() <= 0){
        cum_shape.push_back(shape.begin()[0]);
        return cum_shape;
    }
    cum_shape[cum_shape.size()-1]=shape[shape.size()-offset-1];

    for(int i= (int)cum_shape.size()-1; i>0;i--){
    cum_shape[i-1] = cum_shape[i]*shape[i];
    }

    return cum_shape;
}


ndarray::Shape ndarray::arrayutil::getMatmulOutShape(const ndarray::Shape &l_shape, const ndarray::Shape &r_shape){
    if(l_shape.end()[-1] != r_shape.end()[-2]){
        std::stringstream ss;
        ss<<"("<<ndarray::getVectorIntInString(l_shape)<<") (";
        ss<<ndarray::getVectorIntInString(r_shape)<<")";
        throw ndarray::exception::InvalidShapeException("matmul", ss.str());
    }

    std::vector<int>ret_shape;
    int offset = 2;
    ret_shape = ndarray::broadcast::getBroadCastedShape(l_shape,r_shape, offset);

    if(ret_shape.empty()){
        ret_shape.push_back(1);
    }
                                        

    ret_shape.push_back(l_shape.end()[-2]);
    ret_shape.push_back(r_shape.end()[-1]);

    return ret_shape;
}


}//end ndarray namespace