#include "IDW.hpp"
#include <cmath>
#include <algorithm>


IDW::IDW(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power, double offset, int wrapAxis, int wrapSize) 
    : Mapping(data), wrapAxis(wrapAxis), wrapSize(wrapSize) {
    this->knnTree = new KNNTree(this->queriedPoints);
    this->maxNeighbours = maxNeighbours;
    this->power = power;
    this->offset = offset;
}

IDW::~IDW(){
    delete knnTree;
}

inline double IDW::computeDistance(const std::vector<int>& a, const std::vector<int>& b) const {
    double sum = 0.0;
    const size_t size = a.size();
    
    if (wrapAxis >= 0 && wrapAxis < (int)size && wrapSize > 0) {
        // Handle wrapping axis separately for better branch prediction
        for (size_t i = 0; i < size; ++i) {
            double diff;
            if ((int)i == wrapAxis) {
                int directDist = std::abs(a[i] - b[i]);
                int wrapDist = wrapSize - directDist;
                diff = std::min(directDist, wrapDist);
            } else {
                diff = a[i] - b[i];
            }
            sum += diff * diff;
        }
    } else {
        // No wrapping - faster path
        for (size_t i = 0; i < size; ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
    }
    
    return std::sqrt(sum);
}

double IDW::predict(std::vector<int> query) {
    const float EPS = 1e-6f;

    // Get more neighbors for better discontinuity handling
    int searchNeighbors = std::min((int)(maxNeighbours * 2.5), (int)queriedPoints.size());
    std::vector<KNNTree::KDResult> relevantPoints = knnTree->getKNearest(query, searchNeighbors);
    
    if (relevantPoints.empty()) return 0.0;
    
    // Early exit: if exact match found with zero offset
    if (offset == 0) {
        for (const auto& pv : relevantPoints) {
            double dist = computeDistance(query, pv.point);
            if (dist < EPS) return pv.value;
        }
    }
    
    // Pre-allocate to avoid reallocations
    std::vector<std::pair<double, double>> weightedValues;
    weightedValues.reserve(searchNeighbors);
    
    // Compute wrap-aware distances
    double minDist = std::numeric_limits<double>::max();
    double maxDist = 0.0;
    
    for (const auto& pv : relevantPoints) {
        double dist = computeDistance(query, pv.point);
        weightedValues.emplace_back(dist, pv.value);
        minDist = std::min(minDist, dist);
        maxDist = std::max(maxDist, dist);
    }
    
    // Partial sort - only sort top maxNeighbours (faster than full sort)
    int useCount = std::min((int)weightedValues.size(), maxNeighbours);
    std::partial_sort(weightedValues.begin(), 
                      weightedValues.begin() + useCount, 
                      weightedValues.end());
    
    // Adaptive power based on distance variation
    double distRange = maxDist - minDist;
    bool useAdaptivePower = (distRange > minDist * 1.5);
    int adaptivePower = useAdaptivePower ? (power + 1) : power;
    
    // Accumulate weights
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < useCount; ++i) {
        double dist = weightedValues[i].first;
        double value = weightedValues[i].second;
        
        double weight = 1.0;
        if (dist > offset) {
            weight = 1.0 / std::pow(dist, adaptivePower);
        }
        
        numerator += weight * value;
        denominator += weight;
    }
    
    return denominator > 0 ? numerator / denominator : 0.0;
}

