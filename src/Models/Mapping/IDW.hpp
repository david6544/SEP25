#pragma once

#include <vector>
#include "Mapping.hpp"
#include "../Tools/KNNAlgorithm.hpp"

class IDW : public Mapping {
private:
    KNNTree* knnTree;
    int maxNeighbours, power;

public:
    IDW(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power);

    ~IDW();

    double predict(std::vector<int> query) override;
};