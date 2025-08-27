/**
 * @file StateSpace.hpp
 * @author your name (you@domain.com) PLS UPDATE
 * @brief Implements a StateSpace class that allows for interactions with an n-dimensional integer indexed array.

 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef STATESPACE_H
#define STATESPACE_H
#include <vector>
#include <iostream>
#include <cmath>

class StateSpace {
private:
    std::vector<double> stateSpace;
    const int dimensions;
    const int dimensionSize;

    int coords_to_index(const std::vector<int>& coords) const;

public:
    StateSpace(int dimensions, int dimensionSize, double initialValues = 0.0);

    int get_dimensions() const;

    int get_dimension_size() const;
    
    double get(const std::vector<int>& coords) const;

    std::vector<double> get_raw_representation() const;

    double set(const std::vector<int>& coords, double value);
    
};

#endif // STATESPACE_H