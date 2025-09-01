#include "InputOutput.hpp"

#include <stdexcept>

InputOutput* InputOutput::instance = nullptr;
InputOutput::InputOutput() {
    if (instance == nullptr)
        instance = this;
}
InputOutput* InputOutput::get_instance(){ 
    if (instance == nullptr){
        throw std::runtime_error("IO instance is null - must be set before usage");
    }
    return instance; 
}