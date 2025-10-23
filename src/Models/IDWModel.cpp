#if defined(IDW_MODEL) || defined(TESTING)
#include "IDWModel.hpp"
#include <algorithm>
#include <random>

IDWModel::IDWModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) {

    this->stateSpace = new KDTreeStateSpace(dimensions, dimensionSize);
    this->evenQueries.reserve(totalQueries);

    // Number of evenly spaced points per dimension
    int pointsPerDim = static_cast<int>(std::round(std::pow(totalQueries, 1.0 / dimensions)));
    double step = (dimensionSize - 1.0) / std::max(1, pointsPerDim - 1);

    // Build the grid recursively or by counting up in base pointsPerDim
    for (int i = 0; i < std::pow(pointsPerDim, dimensions) && (int)evenQueries.size() < totalQueries; ++i) {
        std::vector<int> query(dimensions);
        int idx = i;
        for (int d = 0; d < dimensions; ++d) {
            int coordIndex = idx % pointsPerDim;
            idx /= pointsPerDim;
            query[d] = static_cast<int>(std::round(coordIndex * step));
        }
        evenQueries.push_back(query);
    }

    // Optional: randomize to avoid spatial bias
    std::shuffle(evenQueries.begin(), evenQueries.end(), std::mt19937{std::random_device{}()});

    // Debug print (optional)
    for (auto &q : evenQueries) {
        std::cout << "Query: ";
        for (auto v : q) std::cout << v << " ";
        std::cout << "\n";
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