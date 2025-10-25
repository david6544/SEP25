#pragma once
#include "Model.hpp"
#include "Querying/QueryTree.hpp"
#include "Mapping/IDW.hpp"

#include <unordered_map>
#include <vector>
#include <functional>

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
    double scaleRatio = 1, invScale = 1;
    int scaledSize = 1;

    int currentQueryIDX;
    std::vector<int> currentQueryPoint;

    std::vector<std::vector<int>> queryWinTotal;
    
    std::vector<std::pair<int, int>> precomputedBounds;

    inline std::vector<int> lowerResolution(const std::vector<int>& rawQuery) const;
    inline std::vector<std::pair<int,int>> get_unscaled_int_bounds(const std::vector<int>& rawQuery) const;
    inline std::vector<int> sample_unscaled_point_from_scaled(const std::vector<int>& scaledCandidate) const;
    inline std::vector<int> unscale_midpoint(const std::vector<int>& scaledQuery) const;
};
