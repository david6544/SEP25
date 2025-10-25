#pragma once

#include <vector>
#include "Mapping.hpp"
#include "../Tools/KNNAlgorithm.hpp"

class LinearStochastic : public Mapping {
private:
    KNNTree* knnTree;
    int maxNeighbours, power; 
    double offset;

public:
    LinearStochastic(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power, double offset = 0);

    ~LinearStochastic();

    double predict(std::vector<int> query) override;
};