#pragma once
#include "Model.hpp"
#include "Querying/QueryTree.hpp"
#include "Mapping/IDW.hpp"

#include <map>
#include <vector>


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
    std::vector<int> currentQueryPoint;
    std::map<std::vector<int>, std::vector<int>> queryWinTotal;
};