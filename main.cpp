#include "array.h"
#include "vector"
#include "iostream"

int main(){

    Shape s = {3};
    double a[] = {1,2,3};

    ndarray::Array A(s, a);
    ndarray::Array B(s, a);

    auto C = A + B;
    

    

    double * cActualData = C.hostData();
    double cExpectedData[] = {2,4,6};
    for(int i =0; i< C.size(); i++){
        std::cout<<cActualData[i]<<" ";
    }

    std::cout<<std::endl;

    return 0;
}