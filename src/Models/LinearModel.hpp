
#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include "Model.hpp"

class LinearModel : public Model {
public:
    LinearModel(int dimensions, int dimensionSize, int queries);
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    void update_prediction_final();
    int find_next_nonzero_ix(double ix);
    int find_prev_nonzero_ix(double ix);
    int find_next_nonzero_afterzero_ix(double ix);
    int find_prev_nonzero_afterzero_ix(double ix);

private:
    int m_maxQueryCount = 0;
};


#endif //LINEAR_MODEL_H