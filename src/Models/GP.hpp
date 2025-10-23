#pragma once
#define GP
#if defined(GP) || defined(TESTING)
#ifndef GP_MODEL_H
#define GP_MODEL_H

#include "Model.hpp"
#include "../StateSpace/ArrayStateSpace.hpp"
#include <memory>
#include <vector>
#include <random>
#include <Eigen/Dense>

#include "util/adam.hpp"
#include "util/gp.hpp"
#include "util/io.hpp"

/**
 * @brief GP model
 */
class GPModel : public Model {
public:
    GPModel(int dimensions, int dimensionSize, int totalQueries);
    ~GPModel();
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;
private:

    int given_ndims ;
    int given_dims_size;
    int given_nqueries;

    std::vector<size_t> dims_queryspace;
    Eigen::MatrixXd bounds_queryspace;

    const size_t ncorners;
    Eigen::MatrixXd points_initial;

    skx::algo::gp::GP gp;

    skx::algo::optim::AdamConfig adam_config{};





};


#endif //GP_MODE_H
#endif // GP || TESTING