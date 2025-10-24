#if defined(IDW_MODEL) || defined(TESTING)
#include "IDWModel.hpp"
#include <algorithm>
#include <random>

IDWModel::IDWModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) {

    this->stateSpace = new KDTreeStateSpace(dimensions, dimensionSize);
    this->evenQueries.clear();
    this->evenQueries.reserve(totalQueries);

    // Ensure we pick enough points per dimension so that pointsPerDim^dimensions >= totalQueries
    int pointsPerDim = static_cast<int>(std::ceil(std::pow((double)totalQueries, 1.0 / dimensions)));
    // compute max combinations safely as integer power
    long long maxCombos = 1;
    for (int i = 0; i < dimensions; ++i) {
        maxCombos *= pointsPerDim;
        if (maxCombos > (long long)totalQueries * 10) break; // avoid huge loops
    }

    // step: avoid divide-by-zero; if pointsPerDim==1 then only coordinate 0 is used
    double denom = std::max(1, pointsPerDim - 1);
    double step = (dimensionSize - 1.0) / denom;

    // Enumerate grid in base pointsPerDim; stop when we've collected exactly totalQueries
    for (long long i = 0; i < maxCombos && (int)evenQueries.size() < totalQueries; ++i) {
        std::vector<int> q(dimensions);
        long long idx = i;
        for (int d = 0; d < dimensions; ++d) {
            int coordIndex = idx % pointsPerDim;
            idx /= pointsPerDim;
            // scale to [0, dimensionSize-1] using step — clamp to be safe
            int coord = static_cast<int>(std::round(coordIndex * step));
            coord = std::max(0, std::min(dimensionSize - 1, coord));
            q[d] = coord;
        }
        evenQueries.push_back(q);
    }

    for (const auto &qq : evenQueries) {
        std::cout << "Query: ";
        for (int v : qq) std::cout << v << " ";
        std::cout << "\n";
    }
    // If we still don't have enough (rare), fill remaining with random valid points
    std::mt19937 rng((unsigned)std::random_device{}());
    std::uniform_int_distribution<int> unif(0, dimensionSize - 1);
    while ((int)evenQueries.size() < totalQueries) {
        std::vector<int> q(dimensions);
        for (int d = 0; d < dimensions; ++d) q[d] = unif(rng);
        evenQueries.push_back(q);
    }

    
    // Shuffle insertion order to avoid spatial bias when building KD-tree
    std::shuffle(evenQueries.begin(), evenQueries.end(), rng);

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