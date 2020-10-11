#include "broadCasted.h"
#include "util.h"
#include <stdexcept> 
#include <sstream>

namespace ndarray
{

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
        return getBroadCastedShape(r_shape, l_shape, offset);
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

BroadCastedProperty getBroadCastedProperty(const ndarray::Shape &out_shape, const ndarray::Array &A, const ndarray::Array &B,const int offset){
    ndarray::Shape l_shape = ndarray::paddedVector(A.shape(), out_shape.size(), 1);
    ndarray::Shape r_shape = ndarray::paddedVector(B.shape(), out_shape.size(), 1);

    ndarray::Stride l_stride = ndarray::paddedVector(A.stride(), out_shape.size(), 0);
    ndarray::Stride r_stride = ndarray::paddedVector(B.stride(), out_shape.size(), 0);
    

    ndarray::Stride broad_l_stride;
    ndarray::Stride broad_r_stride;

    //keep same stride between (last index to last-offset index).
    int unchanged_stride_index = out_shape.size()-offset;

    for(int i=0;i<out_shape.size();i++){
        int l = l_shape[i];
        int r = r_shape[i];

        int ls = (l==r || r ==1 || i>=unchanged_stride_index) ? l_stride[i] : 0;
        int rs = (l==r || l ==1 || i>=unchanged_stride_index) ? r_stride[i] : 0;

        broad_l_stride.push_back(ls);
        broad_r_stride.push_back(rs);
    } 
    
    return BroadCastedProperty(l_shape, r_shape, broad_l_stride, broad_r_stride);
}


std::vector<int>paddedVector(const std::vector<int>&vec, const int size, const int pad_value){

    if(size<vec.size()){
        std::stringstream ss;
        ss <<"invalid padded size where expected size "<<size;
        ss<<" is less then give data size "<<vec.size()<<".";
        throw std::runtime_error(ss.str());
    }

    int pad_size = size- vec.size();
    std::vector<int>out_vec;
    out_vec.reserve(size);
    out_vec.resize(pad_size, pad_value);

    out_vec.insert(out_vec.end(), vec.begin(), vec.end());
    return out_vec;
}

}//end ndarray namespace