#ifndef ONE_D_MODEL_HPP
#define ONE_D_MODEL_HPP

#include "Model.hpp"
#include <vector>
#include <random>

// Forward-declare kernel utility
double rbf_kernel(double x1, double x2, double lengthScale=1.0, double variance=1.0);

class OneDModel : public Model {
private:
    std::vector<int> xs;       // observed input points
    std::vector<double> ys;    // observed function values
    int queryCount = 0;

    // Hyperparameters for GP kernel
    double lengthScale = 5.0;
    double variance = 1.0;
    double noise = 1e-6;

    std::mt19937 rng;

public:
    OneDModel(int dimensions, int dimensionSize, int totalQueries);

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;

private:
    void gp_predict(int xStar, double &mean, double &stddev);
    std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                            const std::vector<double> &b);
};

#endif // ONE_D_MODEL_HPP