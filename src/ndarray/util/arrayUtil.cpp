#include "ndarray/util/arrayUtil.h"

namespace ndarray
{
ndarray::LL getNumOfElementByShape(const ndarray::Shape &shape){

    ndarray::LL size = 1;
    for(ndarray:: LL x : shape) size *= x;

    return size;
}


ndarray::Shape getCumulativeMultiShape(const ndarray::Shape &shape, const int offset){
    ndarray::Shape cum_shape(std::max((int)shape.size()-offset-1,0));

    if(shape.size() <=1)return cum_shape;

    if(cum_shape.size() <= 0){
        cum_shape.push_back(shape.begin()[0]);
        return cum_shape;
    }
    cum_shape[cum_shape.size()-1]=shape[shape.size()-offset-1];

    for(int i=cum_shape.size()-1; i>0;i--){
    cum_shape[i-1] = cum_shape[i]*shape[i];
    }

    return cum_shape;
}


}//end ndarray namespace