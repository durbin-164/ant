#include "ndarray/core/slices.h"
#include "ndarray/exception/ndexception.h"
#include <cmath> 
#include <sstream>
#include <iostream>

namespace ndarray
{

ndarray::Shape ndarray::slices::getSliceOutShape(const ndarray::Slices &slices){
    ndarray::Shape out_shape;

    for(size_t i =0; i<slices.size(); i ++){
        int dif = ceil( float( slices[i][1]-slices[i][0] ) /slices[i][2]);
        out_shape.push_back(dif);
        // std::cout<<i<<" "<<dif<< " " <<float(slices[i][1])<<" "<<float(slices[i][0]) <<" "<<slices[i][2] <<std::endl;
    }

    return out_shape;
}


ndarray::Slices ndarray::slices::filledSlices(const ndarray::Shape &in_shape, const ndarray::Slices &slices){
   
    if(in_shape.size()<slices.size()){
        std::stringstream ss;
        ss <<"slices size "<<slices.size();
        ss<<"which is greater than input shape size "<<in_shape.size()<<".";
        throw ndarray::exception::InvalidSizeException(ss.str());
    }

    ndarray::Slices ret_slices;
    
    for (size_t i = 0; i < in_shape.size(); i++){
        if(i>= slices.size()){
            ret_slices.push_back({0,in_shape[i],1});
            continue;
        }

        switch (slices[i].size())
        {
            case 1:
                ret_slices.push_back({slices[i][0], slices[i][0]+1, 1});
                break;
            case 2:
                ret_slices.push_back({slices[i][0], slices[i][1], 1});
                break;
            case 3:
                ret_slices.push_back(slices[i]);
                break;
            
            default:
                break;
        }
    }

    return ret_slices;
}



ndarray::LL ndarray::slices::getSliceStartIndex(const ndarray::Slices &slices, const ndarray::Stride &stride){
    ndarray:: LL start_index = 0;
    for(size_t i =0; i<slices.size(); i++){
        start_index +=  slices[i][0] * stride[i];
    }
    return start_index;
}

}