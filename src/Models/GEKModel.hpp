#if defined(GEK) || defined(TESTING)
#ifndef GEK_MODEL_H
#define GEK_MODEL_H

#include "Model.hpp"
#include "../StateSpace/ArrayStateSpace.hpp"
#include <memory>
#include <vector>
#include <random>
#include <Eigen/Dense>

/**
 * @brief The GEKModel is for implementing Gradient-ehanced Kriging,
 *  which is a form of guassian process regression, 
 * this will try to be enhanced through an observation matrix
 */
class GEKModel : public Model {
    std::unique_ptr<ArrayStateSpace> stateSpace;
    std::vector<std::vector<int>> observed_queries;
    std::vector<double> observed_values;
    Eigen::MatrixXd C_inv;
    double theta = 0.01;
    bool trained = false;
    std::mt19937 gen;
    // local neighbourhood settings
    bool use_local_neighborhood = true;
    int local_k = 128; // number of nearest observed points to use for local solves
    // cylindrical/periodic dimension settings
    std::vector<bool> periodic_dims; // which dimensions have periodic boundary conditions
public:
    GEKModel(int dimensions, int dimensionSize, int totalQueries);
    ~GEKModel();
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;
    
    // Configure which dimensions are periodic (for cylindrical state spaces)
    void set_periodic_dimensions(const std::vector<bool>& periodic);
private:
    double covariance(const std::vector<int>& x, const std::vector<int>& y);
    double compute_periodic_distance_squared(const std::vector<int>& x, const std::vector<int>& y);
    void train();
    std::vector<int> get_random_query();
    double get_variance(const std::vector<int>& query);
};


#endif //GEK_MODE_H
#endif // GEK || TESTING