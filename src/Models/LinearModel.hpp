/**
 * @file LinearModel.hpp
 * @author your name (you@domain.com) PLS UPDATE
 * @brief This file declares a LinearModel class that performs linear interpolation on a 1D state space.
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
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