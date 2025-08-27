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
#ifndef DUMB_MODEL_H
#define DUMB_MODEL_H

#include "Model.hpp"

class DumbModel : public Model {
public:
    DumbModel(int dimensions, int dimensionSize);
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
};


#endif //DUMB_MODEL_H