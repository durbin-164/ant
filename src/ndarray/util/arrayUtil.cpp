#include "ndarray/util/arrayUtil.h"
#include <stdexcept> 
#include <sstream>
#include "ndarray/util/util.h"
#include "ndarray/core/broadCasted.h"
#include <iostream>
#include "ndarray/exception/ndexception.h"
#include "ndarray/core/array.h"

namespace ndarray
{
ndarray::LL ndarray::arrayutil::getNumOfElementByShape(const ndarray::Shape &shape){

    ndarray::LL size = 1;
    for(ndarray:: LL x : shape) size *= x;

    return size;
}


ndarray::Shape ndarray::arrayutil::getCumulativeMultiShape(const ndarray::Shape &shape, const int offset){
    ndarray::Shape cum_shape(std::max((int)shape.size()-offset,0));

    if(shape.size()-offset <=0)return cum_shape;

    if(cum_shape.empty()){
        cum_shape.push_back(shape.begin()[0]);
        return cum_shape;
    }
    int cum_size = 1;

    for(int i= (int)cum_shape.size()-1; i>=0;i--){
        cum_shape[i] = cum_size;
        cum_size =  cum_shape[i]*shape[i];
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


ndarray::LL ndarray::arrayutil::getIndexFromIndices(const ndarray::Indices & indices, const ndarray::Array &A){
    if(indices.size() != A.shape().size()){
        std::stringstream ss;
        ss<<"invalid index. index size must be "<<A.shape().size()<<" .";
        throw ndarray::exception::InvalidSizeException(ss.str());
    }

    ndarray::LL index = 0;

    for(size_t i =0; i< A.shape().size(); i++){
        if(indices[i]<0 || indices[i]>=A.shape()[i]){
            std::stringstream ss;
            ss<<"index must be between 0 and "<<A.shape()[i]-1 <<" at axis "<<i<<".";
            throw ndarray::exception::IndexOutOfRangeException(ss.str());
        }

        index += indices[i]*A.stride()[i];
    }

    return index;
}


}//end ndarray namespace