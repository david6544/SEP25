<<<<<<< HEAD
#if defined(LINEAR) || defined(TESTING)
=======
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
>>>>>>> 56afda14851d0a3f0f0d71e1299d472e97b71540
#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include "Model.hpp"
#include "../StateSpace/ArrayStateSpace.hpp"

class LinearModel : public Model {
    ArrayStateSpace* stateSpace;
public:
    LinearModel(int dimensions, int dimensionSize, int totalQueries);
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;
private:
    void update_prediction_final();
    int find_next_nonzero_ix(double ix);
    int find_prev_nonzero_ix(double ix);
    int find_next_nonzero_afterzero_ix(double ix);
    int find_prev_nonzero_afterzero_ix(double ix);
};


#endif //LINEAR_MODEL_H
#endif // LINEAR || TESTING