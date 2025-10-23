#pragma once

#include <vector>
#include "Mapping.hpp"
#include "../Tools/KNNAlgorithm.hpp"

class IDW : public Mapping {
private:
    KNNTree* knnTree;
    int maxNeighbours, power, offset;

public:
    IDW(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power, int offset = 0);

    ~IDW();

    double predict(std::vector<int> query) override;
};