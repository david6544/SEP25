#ifndef RBF_MODEL_HPP
#define RBF_MODEL_HPP

#include "Model.hpp"
#include <vector>
#include <utility>

class RBFModel : public Model {
public:
    RBFModel(int dimensions, int dimensionSize);

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;

    //estimate function value at a point (interpolation)
    double predict(const std::vector<int> &point) const;

    void update_prediction_final();

private:
    double m_sigma; //kernel width
    std::vector<std::pair<std::vector<int>, double>> m_data; // queries + results

    //helper to enumerate all coordinates in the state space
    std::vector<std::vector<int>> generate_all_points() const;

    bool is_queried(const std::vector<int>& point) const;
};

#endif // RBF_MODEL_HPP
