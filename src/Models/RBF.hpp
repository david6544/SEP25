#if defined(RBF) || defined(TESTING)
#ifndef RBF_MODEL_H
#define RBF_MODEL_H

#include "Model.hpp"
#include "../StateSpace/KDTreeStateSpace.hpp"
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>

class RBFModel : public Model {
public:
    enum KernelType { GAUSSIAN, MQ, IMQ, TPS };

    RBFModel(int dimensions, int dimensionSize, int totalQueries);

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int>& query, double result) override;
    double get_value_at(const std::vector<int>& query) override;

private:
    // --- Core storage ---
    KDTreeStateSpace* stateSpace;
    std::vector<std::vector<int>> sample_coords;  // size N x D
    std::vector<double> sample_values;            // size N
    std::vector<double> weights;                  // size N (after training)

    // --- RBF params ---
    KernelType kernel = GAUSSIAN;
    double epsilon = 1.0;     // shape parameter for Gaussian/MQ/IMQ
    double lambda  = 0.0;     // tiny ridge if needed for stability (e.g., 1e-10)

    // --- Sampling helpers ---
    // For dense coverage we generate evenly spaced indices; for sparse we do farthest-point.
    std::vector<int> dense_uniform_query();
    std::vector<int> farthest_point_query();

    // --- Training / selection ---
    void select_kernel_and_params();  // decide kernel, epsilon from D, rho, d_nn
    void compute_global_weights();    // build Phi and solve (Phi + lambda I) w = f

    // --- Math helpers ---
    static double euclid(const std::vector<int>& a, const std::vector<int>& b);
    static double median_1nn_distance(const std::vector<std::vector<int>>& X);
    double phi(double r) const;

    // Dense solver (Gaussian elimination with partial pivoting)
    static bool solve_linear_system(std::vector<std::vector<double>>& A,
                                    std::vector<double>& b,
                                    std::vector<double>& x);

    // Budget / regime helpers
    bool is_dense_sampling() const;          // uses rho threshold
    double budget_ratio() const;             // N / K^D
    double pow_int(double base, int exp) const;

    // Utilities
    bool trained() const { return !weights.empty(); }
};

#endif // RBF_MODEL_H
#endif // RBF || TESTING
