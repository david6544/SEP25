/**
 * @file Model.hpp
 * @author your name (you@domain.com) PLS UPDATE
 * @brief This file declares an abstract Model class that serves as a base for different modeling strategies.
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef MODEL_H
#define MODEL_H
#include <vector>

#include "../StateSpace/StateSpace.hpp"

class Model {
protected:
    StateSpace* stateSpace = nullptr;
public:
        Model(int dimensions, int dimensionSize): stateSpace(new StateSpace(dimensions, dimensionSize)) {};
        virtual std::vector<int> get_next_query() = 0;
        virtual void update_prediction(const std::vector<int> &query, double result) = 0;
        StateSpace get_state_space() const { return *stateSpace; };
};


#endif // MODEL_H