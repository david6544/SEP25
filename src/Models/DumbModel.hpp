<<<<<<< HEAD
#if defined(DUMB) || defined(TESTING)
=======
/**
 * @file DumbModel.hpp
 * @author your name (you@domain.com) PLZ UPDATE
 * @brief The DumbModel is merely for testing the client - It randomly picks query points and sets statespace[query point] = returned value
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
>>>>>>> 56afda14851d0a3f0f0d71e1299d472e97b71540
#ifndef DUMB_MODEL_H
#define DUMB_MODEL_H

#include "Model.hpp"
#include "../StateSpace/ArrayStateSpace.hpp"

class DumbModel : public Model {
    ArrayStateSpace* stateSpace;
public:
    DumbModel(int dimensions, int dimensionSize, int totalQueries);
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;
};


#endif // DUMB_MODEL_H
#endif // DUMB || TESTING