#pragma once
#include <exception>
#include <string>

namespace ndarray::exception
{
class InvalidShapeException: public std::exception
{

   public:
      std::string message;
      explicit InvalidShapeException(const std::string &message_)
      :message(message_){
      }

      InvalidShapeException(const std::string &operation_name, const std::string &shape){
         message = operation_name+" operation could not possible with shape "+shape+".";
      }

      const char * what() const throw(){
         return message.c_str();
      }
   
};

class InvalidSizeException: public std::exception
{

   public:
      std::string message;
      explicit InvalidSizeException(const std::string &message_)
      :message(message_){
      }

      const char * what() const throw(){
         return message.c_str();
      }
   
};


}//end exception namespace