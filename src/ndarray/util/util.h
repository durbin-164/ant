#pragma once
#include <sstream>
#include <string>
#include <vector>

namespace ndarray
{

/**
 * \brief get comma seperated string from a vector int type data.<br/>
 * Example : vec{3,5,6,1,2} => out: "3,5,6,1,2"
 * @param vec int type vector data.
 * @return comma seperated string data of give vector.
 */
std::string getVectorIntInString(const std::vector<int> &vec);
void printVec(const std::vector<int>&vec);

}//end ndaarry namespace