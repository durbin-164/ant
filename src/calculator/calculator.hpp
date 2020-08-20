#pragma once
#include <arrayfire.h>

namespace ant{
    struct Calculator
    {
        /* data */
        af::array a;
        af::array b;
        basic(af::array _a, af::array _b);

        af:: array sum();

        af:: array add(af::array a, af::array b);
        af:: array neg(af::array a);
    };
    

}