#include "IDW.hpp"


IDW::IDW(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power) : Mapping(data){
    this->knnTree = new KNNTree(this->queriedPoints);
    this->maxNeighbours = maxNeighbours;
    this->power = power;
}

IDW::~IDW(){
    delete knnTree;
}

double IDW::predict(std::vector<int> query) {
    size_t dim = query.size();
    float numerator = 0.0f;
    float denominator = 0.0f;
    const float EPS = 1e-6f; // avoid division by zero

    std::vector<KNNTree::KDResult> relevantPoints = knnTree->getKNearest(query, this->maxNeighbours);
    
    for (const auto& pv : relevantPoints) {
        const std::vector<int>& coords = pv.point;
        
        if (pv.distance < EPS || pv.point == query) return pv.value;

        float weight = 1.0f / std::pow(pv.distance, this->power);
        numerator += weight * pv.value;
        denominator += weight;
    }
    
    return numerator / denominator;
}

