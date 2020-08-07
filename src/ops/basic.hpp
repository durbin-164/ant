#pragma once
#include <arrayfire.h>

namespace ant{
    struct basic
    {
        /* data */
        af::array a;
        af::array b;
        basic(af::array _a, af::array _b);

        af:: array sum();
    };
    

}