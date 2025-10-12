#include "TestModel.hpp"

#include <iostream>

TestModel::TestModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) 
{
    qt = new QueryTree(dimensions, dimensionSize, 15);
}

TestModel::~TestModel(){
    if (this->maper) delete this->maper;
    if (this->qt) delete this->qt;
}

std::vector<int> TestModel::get_next_query() {
    std::vector<int> candidate;
    candidate = qt->get_next_query();
    return candidate;
}

void TestModel::update_prediction(const std::vector<int> &query, double result) {
    currentQuery++;
    qt->update_prediction(query, result);
    data.emplace_back(result, query);

    if (currentQuery == totalQueries) {
        maper = new IDW(data, 15, 2);
    }
}

double TestModel::get_value_at(const std::vector<int> &query) {
    if (maper) return maper->predict(query);
    return 0.0;
}
