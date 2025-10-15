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
#include <unordered_set>
#include <sstream>

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

    // --- TPS affine term ---
    std::vector<double> tps_affine;               // (D+1) coefficients for TPS

    // --- RBF params ---
    KernelType kernel = MQ;   // MQ as safer default
    double epsilon = 1.0;
    double lambda  = 1e-8;

    // --- Deduplication ---
    std::unordered_set<std::string> seen_keys;
    static std::string coords_key(const std::vector<int>& c);

    // --- Sampling helpers ---
    std::vector<int> dense_uniform_query();
    std::vector<int> farthest_point_query();

    // --- Training / selection ---
    void select_kernel_and_params();
    void compute_global_weights();

    // --- Math helpers ---
    static double euclid_scaled(const std::vector<int>& a, const std::vector<int>& b, int K);
    static double median_1nn_distance(const std::vector<std::vector<int>>& X, int K);
    double phi(double r) const;

    static bool solve_linear_system(std::vector<std::vector<double>>& A,
                                    std::vector<double>& b,
                                    std::vector<double>& x);

    bool is_dense_sampling() const;
    double budget_ratio() const;
    double pow_int(double base, int exp) const;

    bool trained() const { return !weights.empty(); }
};

#endif // RBF_MODEL_H
#endif // RBF || TESTING
