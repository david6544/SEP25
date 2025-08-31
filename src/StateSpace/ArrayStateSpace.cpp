#include "ArrayStateSpace.hpp"

#include <string>

int ArrayStateSpace::coords_to_index(const std::vector<int>& coords) const {
    if (coords.size() != this->dimensions) 
        throw std::invalid_argument("Index has Invalid Number of Dimensions");
    int rawIndex = 0;
    int multiplier = 1;
    for (int i = dimensions - 1; i > -1; i--) {
        if (coords[i] < 0 || coords[i] >= this->dimensionSize) 
            throw std::out_of_range("Coordinate at index [" + std::to_string(i) + "] out of range");
        rawIndex += coords[i] * multiplier;
        multiplier *= dimensionSize;
    }
    return rawIndex;
}

ArrayStateSpace::ArrayStateSpace(int dimensions, int dimensionSize, double initialValues) :
        StateSpace(dimensions, dimensionSize),
        stateSpaceArray(std::pow(dimensionSize, dimensions), initialValues)
    {}


int ArrayStateSpace::get_dimensions() const {
    return this->dimensions;
}

int ArrayStateSpace::get_dimension_size() const {
    return this->dimensionSize;
}

double ArrayStateSpace::get(const std::vector<int>& coords) const {
    return this->stateSpaceArray[coords_to_index(coords)];
}

std::vector<double> ArrayStateSpace::get_raw_representation() const {
    return this->stateSpaceArray;
}

double ArrayStateSpace::set(const std::vector<int>& coords, double value) {
    return this->stateSpaceArray[coords_to_index(coords)] = value;
}