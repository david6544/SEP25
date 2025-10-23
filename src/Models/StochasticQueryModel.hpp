#pragma once
#include "Model.hpp"
#include "Querying/QueryTree.hpp"
#include "Mapping/IDW.hpp"

#include <unordered_map>
#include <vector>
#include <functional> // for std::hash

// Custom hash function for vector<int>
struct vectorhash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for (auto& i : v) {
            seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class StochasticQueryModel : public Model {
public:
    StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries);
    ~StochasticQueryModel();
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int>& query, double result) override;
    double get_value_at(const std::vector<int>& query) override;
private:
    QueryTree* qt;
    IDW* maper = nullptr;
    std::vector<std::pair<double, std::vector<int>>> data;
    int currentQuery = 0;
    int totalPoints = 0;
    int repeatQueries = 0, shouldQuery = 0;
    int wCount = 0, tCount = 0;
    double scaleRatio = 1;
    int scaledSize = 1;
    std::vector<int> currentQueryPoint;

    // Change from std::map to std::unordered_map using vectorhash
    std::unordered_map<std::vector<int>, std::vector<int>, vectorhash> queryWinTotal;

    std::vector<int> lowerResolution(const std::vector<int>& rawQuery);
    std::vector<int> unscaleCandidate(const std::vector<int>& scaledCandidate);
};
