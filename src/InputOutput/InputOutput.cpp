#include "InputOutput.hpp"

#include <stdexcept>

InputOutput* InputOutput::instance = nullptr;

InputOutput::InputOutput() {
    if (instance == nullptr)
        instance = this;
}

std::vector<int> InputOutput::index_to_coords(int index, int dimensions, int dimensionSize) {
    std::vector<int> coords(dimensions);
    for (int d = dimensions - 1; d >= 0; --d) {
        coords[d] = index % dimensionSize;
        index /= dimensionSize;
    }
    if (index != 0) {
        throw std::out_of_range("index too large for given dimensions");
    }
    return coords;
}

InputOutput* InputOutput::get_instance(){ 
    if (instance == nullptr){
        throw std::runtime_error("IO instance is null - must be set before usage");
    }
    return instance; 
}