#include "math.hpp"
#include <arrayfire.h>

using namespace af;

int add(int i, int j){
   array zeros = randu(1, 3);
    return i + j;
}

int subtract(int i, int j){
    return i - j;
}

