#include "util.h"

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

}//end ndaarry namespace