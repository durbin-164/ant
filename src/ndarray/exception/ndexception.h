#pragma once
#include <exception>
#include <string>

namespace ndarray::exception
{
class InvalidShapeException: public std::exception
{

   public:
      std::string message;
      InvalidShapeException(std::string message_)
      :message(message_){
      }

      InvalidShapeException(std::string operation_name, std::string shape){
         message = operation_name+" operation could not possible with shape "+shape+".";
      }

      std::string what(){
         return message;
      }
   
};

class InvalidSizeException: public std::exception
{

   public:
      std::string message;
      InvalidSizeException(std::string message_)
      :message(message_){
      }

      std::string what(){
         return message;
      }
   
};


}//end exception namespace