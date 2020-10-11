#pragma once
#include "array.h"

namespace cuda
{

    ndarray::Array cudaAdd(const ndarray::Array &A, const ndarray::Array &B);
    ndarray::Array cudaMatmul(const ndarray::Array &A, const ndarray::Array &B);

}