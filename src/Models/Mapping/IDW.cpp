#include "IDW.hpp"
#include <cmath>
#include <limits>

IDW::IDW(std::vector<std::pair<double, std::vector<int>>>& data,
         int maxNeighbours,
         double power,
         double offset)
    : Mapping(data),
      maxNeighbours(maxNeighbours),
      power(power),
      offset(offset) 
{
    this->knnTree = new KNNTree(this->queriedPoints);
}

IDW::~IDW() {
    delete knnTree;
}

double IDW::predict(std::vector<int> query) {
    constexpr double EPS = 1e-9; // prevent divide-by-zero
    double numerator = 0.0;
    double denominator = 0.0;

    auto relevantPoints = knnTree->getKNearest(query, this->maxNeighbours);
    if (relevantPoints.empty()) return 0.0;

    double localDensity = relevantPoints.back().distance / relevantPoints.front().distance;
    double adaptivePower = this->power * std::log1p(localDensity);


    for (const auto& pv : relevantPoints) {
        // If query exactly matches a known point, return its value directly
        if (pv.distance < EPS)
            return pv.value;

        // Efficient weight calculation
        double d = std::max(pv.distance, offset);
        double weight = 1.0 / std::pow(d, adaptivePower);

        numerator += weight * pv.value;
        denominator += weight;
    }

    // Avoid division by zero (shouldn’t happen, but just in case)
    if (denominator < EPS)
        return 0.0;

    return numerator / denominator;
}
