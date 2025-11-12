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
#if defined(DUMB) || defined(TESTING)
#include <vector>
#include <random>
#include <ctime>

#include "DumbModel.hpp"

DumbModel::DumbModel(int dimensions, int dimensionSize, int totalQueries) : 
    Model(dimensions, dimensionSize, totalQueries), stateSpace(new ArrayStateSpace(dimensions, dimensionSize)) {
    std::srand(std::time(nullptr));
}

std::vector<int> DumbModel::get_next_query() {
    return {0,0,0};
}

void DumbModel::update_prediction(const std::vector<int> &query, double result) {
    return;
}

double DumbModel::get_value_at(const std::vector<int> &query) {
    return 0.335;
}

#endif // DUMB || TESTING