#if defined(RBF) || defined(TESTING)
#ifndef RBF_MODEL_H
#define RBF_MODEL_H

#include "Model.hpp"
#include "../StateSpace/ArrayStateSpace.hpp"
#include <vector>
#include <utility>

class RBFModel : public Model {
    ArrayStateSpace* stateSpace;

public:
    RBFModel(int dimensions, int dimensionSize, int totalQueries);

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;

private:
    double m_sigma;
    std::vector<std::pair<std::vector<int>, double>> m_data;

    std::vector<std::vector<int>> generate_all_points() const;
    bool is_queried(const std::vector<int>& point) const;
    double predict(const std::vector<int> &point) const;
    void update_prediction_final();
};

#endif // RBF_MODEL_H
#endif // RBF || TESTING
