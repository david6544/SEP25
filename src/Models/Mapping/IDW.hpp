#pragma once

#include <vector>
#include <limits>
#include "Mapping.hpp"
#include "../Tools/KNNAlgorithm.hpp"

class IDW : public Mapping {
private:
    KNNTree* knnTree;
    int maxNeighbours, power; 
    double offset;
    int wrapAxis;           // Which axis wraps (-1 for none)
    int wrapSize;           // Size of the wrapping dimension

public:
    IDW(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power, double offset = 0, int wrapAxis = -1, int wrapSize = 0);

    ~IDW();

    double predict(std::vector<int> query) override;
    
private:
    inline double computeDistance(const std::vector<int>& a, const std::vector<int>& b) const;
};