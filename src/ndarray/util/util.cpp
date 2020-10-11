#include "util.h"
#include <stdio.h>
#include "cuda.h"
#include <cuda_runtime.h>

namespace ndarray
{

std::string getVectorIntInString(const std::vector<int> &vec){

    std::stringstream ss;

    if(vec.size()) ss <<vec[0];

    for(size_t i = 1; i < vec.size(); i++){
        ss <<","<<vec[i];

    }

    return ss.str();
}

// void printVec(const std::vector<int>vec){
//   for(int x: vec)printf("%d ", x);
//   printf("\n");
// }

// void printInt(const int * data, int N){
//   int *hdata = (int*)malloc(N*sizeof(int));
//   cudaMemcpy(hdata, data, N*sizeof(int), cudaMemcpyDeviceToHost);
//   for(int i =0 ; i < N; i++)printf("%d ", hdata[i]);
//   printf("\n");
// }

// void printDouble(const double * data, int N){
//   double *hdata = (double*)malloc(N*sizeof(double));
//   cudaMemcpy(hdata, data, N*sizeof(double), cudaMemcpyDeviceToHost);
//   for(int i =0 ; i < N; i++)printf("%lf ", hdata[i]);
//   printf("\n");
// }


}//end ndaarry namespace