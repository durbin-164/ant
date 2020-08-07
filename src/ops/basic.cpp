#include "basic.hpp"
#include <arrayfire.h>

// using ant::basic;
namespace ant{
    basic::basic(af::array _a, af::array _b){
    basic::a = std::move(_a);
    basic::b = std::move(_b);
    }

    af:: array basic::sum(){
    return a + b;
}





}
