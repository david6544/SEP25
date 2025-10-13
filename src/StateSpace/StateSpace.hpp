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

#ifndef STATE_SPACE_H
#define STATE_SPACE_H
#include <vector>
#include <iostream>
#include <cmath>

class StateSpace {
protected:
    int dimensions;
    int dimensionSize;
public:
    StateSpace(int dimensions, int dimensionSize): dimensions(dimensions), dimensionSize(dimensionSize) {}
    
    virtual int get_dimensions() const = 0;

    virtual int get_dimension_size() const = 0;
    
    virtual double get(const std::vector<int>& coords) const = 0;
};

#endif // STATE_SPACE_H