#include "OneDModel.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

// ===== Utility function =====
double rbf_kernel(double x1, double x2, double lengthScale, double variance) {
    double d = (x1 - x2) / lengthScale;
    return variance * std::exp(-0.5 * d * d);
}

// ===== Constructor =====
OneDModel::OneDModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) 
{
    if (dimensions != 1) {
        throw std::invalid_argument("OneDModel only supports 1D");
    }
    std::random_device rd;
    rng.seed(rd());
}

// ===== Main interface =====
std::vector<int> OneDModel::get_next_query() {
    if (queryCount == 0) {
        queryCount++;
        return {dimensionSize / 2}; // start from the middle
    }

    int bestX = 0;
    double bestAcq = -1e18;

    // Scan domain for maximum Expected Improvement (EI)
    for (int x = 0; x < dimensionSize; x++) {
        double mean, stddev;
        gp_predict(x, mean, stddev);

        double fBest = *std::max_element(ys.begin(), ys.end());
        double improvement = mean - fBest;
        double z = (stddev > 1e-9) ? improvement / stddev : 0.0;

        // Expected Improvement (EI)
        double ei = (improvement > 0 ? improvement : 0)
                  * std::erfc(-z / std::sqrt(2)) / 2
                  + stddev * (1.0 / std::sqrt(2*M_PI)) * std::exp(-0.5 * z * z);

        if (ei > bestAcq) {
            bestAcq = ei;
            bestX = x;
        }
    }

    queryCount++;
    return {bestX};
}

void OneDModel::update_prediction(const std::vector<int> &query, double result) {
    if (query.size() != 1) throw std::invalid_argument("Expected 1D query");
    xs.push_back(query[0]);
    ys.push_back(result);
}

double OneDModel::get_value_at(const std::vector<int> &query) {
    if (query.size() != 1) throw std::invalid_argument("Expected 1D query");
    double mean, stddev;
    gp_predict(query[0], mean, stddev);
    return mean;
}

// ===== Gaussian Process prediction =====
void OneDModel::gp_predict(int xStar, double &mean, double &stddev) {
    int n = xs.size();
    if (n == 0) {
        mean = 0.0;
        stddev = 1.0;
        return;
    }

    // Build covariance matrix K (n x n)
    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            K[i][j] = rbf_kernel(xs[i], xs[j], lengthScale, variance);
        }
        K[i][i] += noise; // add noise to diagonal
    }

    // Compute k* (covariance between x* and training points)
    std::vector<double> kStar(n);
    for (int i = 0; i < n; i++) {
        kStar[i] = rbf_kernel(xStar, xs[i], lengthScale, variance);
    }

    // Solve for alpha = K^-1 y
    std::vector<double> alpha = solve_linear_system(K, ys);

    // Predictive mean
    mean = 0.0;
    for (int i = 0; i < n; i++) mean += kStar[i] * alpha[i];

    // Predictive variance
    double kStarStar = rbf_kernel(xStar, xStar, lengthScale, variance) + noise;
    std::vector<double> v = solve_linear_system(K, kStar);
    double kTK = 0.0;
    for (int i = 0; i < n; i++) kTK += kStar[i] * v[i];

    double var = kStarStar - kTK;
    stddev = (var > 0) ? std::sqrt(var) : 1e-6;
}

// ===== Linear system solver (Gaussian elimination) =====
std::vector<double> OneDModel::solve_linear_system(std::vector<std::vector<double>> A,
                                                      const std::vector<double> &b) {
    int n = A.size();
    std::vector<std::vector<double>> aug(n, std::vector<double>(n+1, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
        aug[i][n] = b[i];
    }

    // Gaussian elimination
    for (int i = 0; i < n; i++) {
        // pivot
        int maxRow = i;
        for (int k = i+1; k < n; k++) {
            if (fabs(aug[k][i]) > fabs(aug[maxRow][i])) maxRow = k;
        }
        std::swap(aug[i], aug[maxRow]);

        // normalize row
        double diag = aug[i][i];
        if (fabs(diag) < 1e-12) continue;
        for (int j = i; j <= n; j++) aug[i][j] /= diag;

        // eliminate
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = aug[k][i];
            for (int j = i; j <= n; j++) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // extract solution
    std::vector<double> x(n);
    for (int i = 0; i < n; i++) x[i] = aug[i][n];
    return x;
}
