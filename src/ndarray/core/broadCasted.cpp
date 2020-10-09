#include "broadCasted.h"
#include "util.h"
#include <stdexcept> 
#include <sstream>

namespace ndarray
{

struct BroadCastedProperty{
    Shape a_shape;
    Shape b_shape;
    Stride a_stride;
    Stride b_stride;

    BroadCastedProperty(Shape &a_shape_, Shape &b_shape_, Stride &a_stride_, Stride &b_stride_ ){
        a_shape = a_shape_;
        b_shape = b_shape_;
        a_stride = a_stride_;
        b_stride = b_stride_;
    }
};


Shape getBroadCastedShape(const Shape &l_shape, const Shape &r_shape,const int offset){
    if(l_shape.size()==0 || (l_shape.size()==1 && l_shape[0]==0) ||
       r_shape.size()==0 || (r_shape.size()==1 && r_shape[0]==0)){
        std::stringstream ss;
        ss<<"operands could not be broadcast together with shapes(";
        ss<<ndarray::getVectorIntInString(l_shape)<<") (";
        ss<<ndarray::getVectorIntInString(r_shape)<<").";
        throw std::runtime_error(ss.str());
    }

    if(l_shape.size()<r_shape.size()){
        return getBroadCastedShape(r_shape, l_shape);
    }


    std::vector<int>out_shape(l_shape.size()-offset);

    size_t r_offset = l_shape.size()-r_shape.size();

    for(size_t i= 0; i<l_shape.size()-offset; i++){
        if(i<r_offset){
            out_shape[i]=l_shape[i];
        }else{
            int l = l_shape[i];
            int r = r_shape[i-r_offset];

            if(l==r){
                out_shape[i]=l; //No broadcast
            }else if(l==1){
                out_shape[i]=r; //right broadcast
            }else if(r==1){
                out_shape[i]=l; //left braodcast
            }else{
                std::stringstream ss;
                ss<<"operands could not be broadcast together with shapes(";
                ss<<ndarray::getVectorIntInString(l_shape)<<") (";
                ss<<ndarray::getVectorIntInString(r_shape)<<").";
                throw std::runtime_error(ss.str());
            }
        }
    }

    return out_shape;
}


}//end ndarray namespace