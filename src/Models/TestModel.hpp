#pragma once
#include "Querying/QueryTree.hpp"
#include "Model.hpp"
#include "Mapping/IDW.hpp"

#include <unordered_set>

class TestModel : public Model {
public:
    TestModel(int dimensions, int dimensionSize, int totalQueries);
    ~TestModel();
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int>& query, double result) override;
    double get_value_at(const std::vector<int>& query) override;

private:
    QueryTree* qt;
    IDW* maper = nullptr;
    std::vector<std::pair<double, std::vector<int>>> data;
    int currentQuery = 0;
};
