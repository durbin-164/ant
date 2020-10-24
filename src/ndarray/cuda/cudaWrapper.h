#pragma once
#include "array.h"

namespace cuda
{

/**
 * \brief add two ndarray in cuda.
 * @param A left ndarray
 * @param B right ndarray
 * @return another ndarray which have A+B data.
 */
ndarray::Array cudaAdd(const ndarray::Array &A, const ndarray::Array &B);

/**
 * \brief matmul of two ndarray in cuda.
 * @param A left ndarray
 * @param B right ndarray
 * @return another ndarray which have A.matmul(B) data.
 */
ndarray::Array cudaMatmul(const ndarray::Array &A, const ndarray::Array &B);

/**
 * \brief slices a ndarray in cuda and return a cuda ndarray.
 * @param A a ndarray
 * @param slices a 2D vector of slices.
 * @return another ndarray with slices value.
 */
ndarray::Array cudaSlices(const ndarray::Array &A, const ndarray::Slices &slices);

}