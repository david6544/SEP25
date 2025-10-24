#pragma once
#if defined(GPMODEL) || defined(TESTING)
#ifndef GP_MODEL_H
#define GP_MODEL_H

#include "Model.hpp"
#include "../StateSpace/ArrayStateSpace.hpp"
#include <memory>
#include <vector>
#include <random>
#include <Eigen/Dense>

#include "../util/adam.hpp"
#include "../util/gp.hpp"
#include "../util/io.hpp"

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

    std::vector<int> query_holder;
    Eigen::RowVectorXd point_sample_next;

    int given_ndims;
    int given_dims_size;
    int given_nqueries;

    std::vector<size_t> dims_queryspace;
    Eigen::MatrixXd bounds_queryspace;

    size_t ncorners;
    Eigen::MatrixXd points_initial;

    skx::algo::gp::GP gp;

    skx::algo::optim::AdamConfig adam_config{};





};


#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>

GPModel::GPModel(int dimensions, int dimensionSize, int totalQueries) :
    Model(dimensions, dimensionSize, totalQueries), query_holder(dimensions)
{
    given_ndims = dimensions;
    given_dims_size = dimensionSize;
    given_nqueries = totalQueries;

    dims_queryspace = std::vector<size_t>(given_ndims);
    for (size_t& d : dims_queryspace) {
        d = given_dims_size;
    }

    bounds_queryspace = Eigen::MatrixXd(given_ndims, 2);
    for (size_t i = 0; i < given_ndims; i++) {
        bounds_queryspace(i, 0) = 0.0;
        bounds_queryspace(i, 1) = static_cast<double>(dims_queryspace[i]-1);
    }

    ncorners = (1ULL << given_ndims);
    points_initial = Eigen::MatrixXd(ncorners + 1, given_ndims);
    for (size_t i = 0; i < ncorners; i++) {
        for (size_t d = 0; d < given_ndims; d++) {
            points_initial(i, d) = static_cast<double>(((i & (1ULL << d)) > 0)) * (bounds_queryspace(d,1)-bounds_queryspace(d,0)) + bounds_queryspace(d,0);
        }
    }
    for (size_t d = 0; d < given_ndims; d++) {
		points_initial(ncorners, d) = std::floor(0.5 * (bounds_queryspace(d,1)-bounds_queryspace(d,0)) + bounds_queryspace(d,0));
    }

    gp = skx::algo::gp::gp_create(given_ndims);
    gp.kernel_func = skx::algo::gp::kernel::ker_sum_rbf_plus_periodic_ard<const Eigen::VectorXd&, const Eigen::RowVectorXd&, const Eigen::RowVectorXd&>;
    gp.kernel_params = skx::algo::gp::kernel::ker_sum_rbf_plus_periodic_ard_init(given_ndims);

    adam_config.params_is_logspace = false;
    adam_config.kernel_func_is_logspace = false;
    adam_config.kernel_grad_func_is_logspace = false;
    adam_config.alpha = 0.1;
}

GPModel::~GPModel() {
}

void fill_vec_with_eigen_vec(std::vector<int>& v, const Eigen::RowVectorXd& eiv) {
    for (size_t i = 0; i < eiv.size(); i++) {
        v[i] = static_cast<int>(eiv(i));
    }
}

std::vector<int> GPModel::get_next_query() {
    if (currentQuery < points_initial.size()) {
        fill_vec_with_eigen_vec(query_holder, points_initial.row(currentQuery));

        return query_holder;
    }
    skx::algo::optim::optimise_params_adam(
        gp.kernel_params, 
        gp.X_sample, 
        gp.Y_sample, 
        gp.var_noise, 
        gp.kernel_func, 
        //skx::algo::gp::kernel::ker_rbf_ard_grad, 
        skx::algo::gp::kernel::ker_sum_rbf_plus_periodic_ard_grad,
        adam_config
    );
    skx::algo::gp::gp_on_update_kernel(gp);
    //if (adam_config.t_max > 2) {
    //    adam_config.t_max += -1;
    //}
    point_sample_next = skx::algo::gp::gp_point_argmax_reduce_IMSE(gp);
    fill_vec_with_eigen_vec(query_holder, point_sample_next);
    return query_holder;
}

void GPModel::update_prediction(const std::vector<int> &query, double response) {
    if (currentQuery < points_initial.size()) {
        skx::algo::gp::gp_add_sample(gp, points_initial.row(currentQuery), static_cast<double>(response));
        return;
    }
    if (points_initial.size() == currentQuery+1) {
        Eigen::VectorXd y_sample = gp.Y_sample.col(0);
        gp.kernel_params = skx::algo::gp::kernel::ker_sum_rbf_plus_periodic_ard_init_from_data(gp.X_sample, &y_sample);
        skx::algo::gp::gp_add_predict_points(gp, skx::algo::gp::enumerate_all_integer_points(bounds_queryspace));

    }

    skx::algo::gp::gp_add_sample(gp, point_sample_next, static_cast<double>(response));

    if (currentQuery-1 == given_nqueries) { 
        adam_config.t_max = 64;
        adam_config.alpha = 0.001;
        skx::algo::optim::optimise_params_adam(
            gp.kernel_params, 
            gp.X_sample, 
            gp.Y_sample, 
            gp.var_noise, 
            gp.kernel_func, 
            //skx::algo::gp::kernel::ker_rbf_ard_grad, 
            skx::algo::gp::kernel::ker_sum_rbf_plus_periodic_ard_grad,
            adam_config
        );
        skx::algo::gp::gp_on_update_kernel(gp);
    }
    currentQuery++;
}

double GPModel::get_value_at(const std::vector<int>& point_queryspace) {
    double expectation = 0.0;
    Eigen::RowVectorXd X_predict_new(given_ndims);
    for (size_t i = 0; i < given_ndims; i++) {
        X_predict_new(i) = static_cast<double>(point_queryspace[i]);
    }
    expectation = skx::algo::gp::gp_calc_expectation(gp, X_predict_new);
    return expectation;
}




#endif //GP_MODE_H
#endif // GP || TESTING