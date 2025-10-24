#if defined(IDW_MODEL) || defined(TESTING)
#include "IDWModel.hpp"
#include <algorithm>
#include <random>

IDWModel::IDWModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) {

    this->stateSpace = new KDTreeStateSpace(dimensions, dimensionSize);
    this->evenQueries.clear();
    this->evenQueries.reserve(totalQueries);


    // Generate evenly distributed queries across the entire state space
    int numPerDim = std::round(std::pow(totalQueries, 1.0 / dimensions));
    if (numPerDim < 1) numPerDim = 1;

    // Step size per dimension (ensure coverage of [0, dimensionSize-1])
    double step = (dimensionSize - 1.0) / std::max(1, numPerDim - 1);

    // Recursive lambda to generate all grid combinations
    std::vector<std::vector<int>> gridPoints;
    std::vector<int> current(dimensions, 0);

    std::function<void(int)> generateGrid = [&](int dim) {
        if (dim == dimensions) {
            gridPoints.push_back(current);
            return;
        }
        for (int i = 0; i < numPerDim; ++i) {
            current[dim] = std::round(i * step);
            generateGrid(dim + 1);
        }
    };

    generateGrid(0);

    // Shuffle and trim if we generated more than totalQueries
    std::shuffle(gridPoints.begin(), gridPoints.end(), std::mt19937(std::random_device{}()));
    if ((int)gridPoints.size() > totalQueries)
        gridPoints.resize(totalQueries);

    evenQueries = std::move(gridPoints);

}

// Safe get_next_query: check evenQueries.size(), not totalQueries
std::vector<int> IDWModel::get_next_query() {
    if (queryIdx < (int)evenQueries.size()) {
        return evenQueries[queryIdx++];
    }
    return std::vector<int>(dimensions, 0); // Fallback (should not happen)
}


void IDWModel::update_prediction(const std::vector<int> &query, double result) {
    stateSpace->insert(query, result);
}

double IDWModel::get_value_at(const std::vector<int> &query) {
    // Use IDW to compute the distance weighted by k nearest neighbors
    const int k = 5;
    auto neighbors = stateSpace->k_nearest_neighbors(query, k);
    if(stateSpace->get(query) != -1.0){
        return stateSpace->get(query);
    }
    if (neighbors.empty()) {
        std::cout << "Neighbors found: " << neighbors.size() << std::endl;
        return 0.0; // No data available
    }
    double weightedSum = 0.0;
    double totalWeight = 0.0;
    for (const auto& neighbor : neighbors) {
        double dist2 = stateSpace->squared_distance(query, neighbor->coords);
        double weight = 1.0 / (dist2 + 1e-6);  // no double squaring

        weightedSum += weight * neighbor->value;
        totalWeight += weight;
    }
    return weightedSum / totalWeight;
}
#endif // IDW || TESTING