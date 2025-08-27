/**
 * @file DumbModel.cpp
 * @author Cuinn Kemp
 * @brief This file implements a simple model that randomly selects query points
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <vector>
#include <random>
#include <ctime>

#include "DumbModel.hpp"

DumbModel::DumbModel(int dimensions, int dimensionSize) : Model(dimensions, dimensionSize) {
    std::srand(std::time(nullptr));
}

std::vector<int> DumbModel::get_next_query() {
    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    std::vector<int> nextQuery(D, 0);

    for (auto &var : nextQuery){
        var = std::rand() % K;
    }

    return nextQuery;
}

void DumbModel::update_prediction(const std::vector<int> &query, double result) {
    stateSpace->set(query, result);
}

