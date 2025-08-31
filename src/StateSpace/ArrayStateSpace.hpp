/* 

ArrayArrayStateSpace.hpp
Implements a ArrayStateSpace class that allows for interactions with an n-dimensional integer indexed array.

*/


#ifndef ARRAY_STATE_SPACE_H
#define ARRAY_STATE_SPACE_H
#include <vector>
#include <iostream>
#include <cmath>

#include "StateSpace.hpp"

class ArrayStateSpace : public StateSpace {
private:
    std::vector<double> stateSpaceArray;
    
    int coords_to_index(const std::vector<int>& coords) const;

public:
    ArrayStateSpace(int dimensions, int dimensionSize, double initialValues = 0.0);

    int get_dimensions() const override;

    int get_dimension_size() const override;
    
    double get(const std::vector<int>& coords) const override;

    std::vector<double> get_raw_representation() const;

    double set(const std::vector<int>& coords, double value);
    
};

#endif // ARRAY_STATE_SPACE_H