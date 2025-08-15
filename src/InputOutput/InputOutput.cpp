#include "InputOutput.hpp"

InputOutput* InputOutput::instance = nullptr;
InputOutput::InputOutput() { instance = this; }
InputOutput* InputOutput::get_instance(){ return instance; }