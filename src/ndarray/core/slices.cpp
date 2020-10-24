#include "ndarray/core/slices.h"
#include "ndarray/exception/ndexception.h"
#include <cmath> 
#include <sstream>
#include <iostream>

namespace ndarray
{
void varifyStartAndEndValue(int &start, int &end, int shape);
ndarray::Slices ndarray::slices::filledSlices(const ndarray::Shape &in_shape, const ndarray::Slices &slices){
   
    if(in_shape.size()<slices.size()){
        std::stringstream ss;
        ss <<"slices size "<<slices.size();
        ss<<" which is greater than input shape size "<<in_shape.size()<<".";
        throw ndarray::exception::InvalidSizeException(ss.str());
    }

    ndarray::Slices ret_slices;
    
    for (size_t i = 0; i < in_shape.size(); i++){
        if(i>= slices.size()){
            ret_slices.push_back({0,in_shape[i],1});
            continue;
        }

        int start;
        int end;
        std::stringstream ss;

        switch (slices[i].size())
        {
            case 1:
                start = slices[i][0];
                if(start<0) start = in_shape[i]+start;
                if(start<0 || start >= in_shape[i]) {
                    ss<<"index "<<slices[i][0]<<" is out of bounds for axis "<<i<<" with size "<<in_shape[i];
                    throw ndarray::exception::IndexOutOfRangeException(ss.str());
                }
                ret_slices.push_back({start, start+1, 1});
                break;
            case 2:
                start =slices[i][0];
                end = slices[i][1];
                ndarray::varifyStartAndEndValue(start, end, in_shape[i]);
                ret_slices.push_back({start, end, 1});
                
                break;
            case 3:
                start =slices[i][0];
                end = slices[i][1];
                ndarray::varifyStartAndEndValue(start, end, in_shape[i]);
                ret_slices.push_back({start, end, slices[i][2]});
                break;
            
            default:
                ss<<"slice parameter size "<<slices[i].size()<<" out of range at axis "<<i;
                throw ndarray::exception::InvalidSizeException(ss.str());
        }
    }

    return ret_slices;
}


ndarray::Shape ndarray::slices::getSliceOutShape(const ndarray::Slices &slices){
    ndarray::Shape out_shape;

    for(auto slice:slices){
        int dif = std::max( (int)ceil( float( slice[1]-slice[0] ) / float(slice[2])), 0);
        out_shape.push_back(dif);
    }

    return out_shape;
}


ndarray::LL ndarray::slices::getSliceStartIndex(const ndarray::Slices &slices, const ndarray::Stride &stride){
    ndarray:: LL start_index = 0;
    for(size_t i =0; i<slices.size(); i++){
        start_index +=  slices[i][0] * stride[i];
    }
    return start_index;
}

void varifyStartAndEndValue(int &start, int &end, int shape){
    
    if(start < 0) start = shape+start;
    if(end < 0) end = shape + end;

    if(start<0) start =0;
    else if(start>shape)start = shape;
    
    if(end<0) end = 0;
    else if(end>shape) end = shape;
}
}