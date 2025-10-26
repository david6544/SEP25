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
                diff = (double)std::min(directDist, wrapDist);
            } else {
                diff = (double)(a[i] - b[i]);
            }
            sum += diff * diff;
        }
    } else {
        // No wrapping - faster path
        for (size_t i = 0; i < size; ++i) {
            double diff = (double)(a[i] - b[i]);
            sum += diff * diff;
        }
    }
    
    return std::sqrt(sum);
}

double IDW::predict(std::vector<int> query) {
    const float EPS = 1e-6f;

    // ULTRA-FAST: Use exactly maxNeighbours, no extra search
    std::vector<KNNTree::KDResult> relevantPoints = knnTree->getKNearest(query, maxNeighbours);
    
    if (relevantPoints.empty()) return 0.0;
    
    // ULTRA-FAST: Skip sorting, use all returned neighbors directly
    int useCount = std::min((int)relevantPoints.size(), maxNeighbours);
    
    // Simplified power - use fixed power for speed
    bool usePower2 = (power == 2);
    bool usePower1 = (power == 1);
    
    // Accumulate weights
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < useCount; ++i) {
        double dist = computeDistance(query, relevantPoints[i].point);
        double value = relevantPoints[i].value;
        
        // Early exit: if exact match found
        if (dist < EPS) return value;
        
        double effectiveDist = std::max(dist, offset);
        double weight;
        
        // Ultra-fast power calculation
        if (usePower1) {
            weight = 1.0 / effectiveDist;
        } else if (usePower2) {
            weight = 1.0 / (effectiveDist * effectiveDist);
        } else {
            weight = 1.0 / std::pow(effectiveDist, power);
        }
        
        numerator += weight * value;
        denominator += weight;
    }
    
    return denominator > 0 ? numerator / denominator : 0.0;
}

