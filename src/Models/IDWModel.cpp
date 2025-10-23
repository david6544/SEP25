#if defined(IDW_MODEL) || defined(TESTING)
#include "IDWModel.hpp"

IDWModel::IDWModel(int dimensions, int dimensionSize, int totalQueries) : Model(dimensions, dimensionSize, totalQueries) {
    this->stateSpace = new KDTreeStateSpace(dimensions, dimensionSize);
    this->evenQueries = std::vector<std::vector<int>>(totalQueries);
    // Precompute even queries
    for (int q = 0; q < totalQueries; ++q) {
        int pointsPerDim = static_cast<int>(std::round(std::pow(totalQueries, 1.0 / dimensions)));
        std::vector<int> query(dimensions);
        int temp = q;
        for (int d = 0; d < dimensions; ++d) {
            query[d] = temp % pointsPerDim;
            temp /= pointsPerDim;
        }
        evenQueries[q] = query;
    }
}


std::vector<int> IDWModel::get_next_query() {
    if (queryIdx < totalQueries) {
        std::vector<int> query = this->evenQueries[queryIdx++];
        return query;
    }

    return std::vector<int>();
}



void IDWModel::update_prediction(const std::vector<int> &query, double result) {
    stateSpace->insert(query, result);
}
double IDWModel::get_value_at(const std::vector<int> &query) {
    // Use IDW to compute the distance weighted by k nearest neighbors
    const int k = 5;
    auto neighbors = stateSpace->k_nearest_neighbors(query, k);
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