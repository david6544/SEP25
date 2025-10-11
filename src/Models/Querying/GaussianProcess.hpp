#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

/**
 * Minimal Gaussian Process implementation using RBF kernel.
 * No external dependencies.
 */
class GaussianProcess {
public:
    GaussianProcess(double lengthScale = 1.0, double signalVariance = 1.0, double noiseVariance = 1e-6)
        : l(lengthScale), sigma_f(signalVariance), sigma_n(noiseVariance) {}

    void fit(const std::vector<std::vector<int>>& X_train,
             const std::vector<double>& y_train)
    {
        X = X_train;
        y = y_train;
        const int n = (int)X.size();

        if (n == 0)
            throw std::invalid_argument("GaussianProcess::fit: No training data given.");

        // Build kernel matrix
        K.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                K[i][j] = kernel(X[i], X[j]);

        // Add noise to diagonal
        for (int i = 0; i < n; ++i)
            K[i][i] += sigma_n;

        // Compute inverse of K
        K_inv = invertMatrix(K);
    }

    // Predict mean and variance at new point
    std::pair<double, double> predict(const std::vector<int>& x_star) const {
        const int n = (int)X.size();
        if (n == 0)
            return {0.0, 1.0}; // fallback if no data

        // Compute k_star
        std::vector<double> k_star(n);
        for (int i = 0; i < n; ++i)
            k_star[i] = kernel(X[i], x_star);

        double k_star_star = kernel(x_star, x_star);

        // mu = k_star^T * K_inv * y
        std::vector<double> Kinv_y = matVecMul(K_inv, y);
        double mu = dot(k_star, Kinv_y);

        // sigma^2 = k_star_star - k_star^T * K_inv * k_star
        std::vector<double> Kinv_k = matVecMul(K_inv, k_star);
        double sigma2 = k_star_star - dot(k_star, Kinv_k);
        if (sigma2 < 1e-12) sigma2 = 1e-12; // numerical guard

        return {mu, std::sqrt(sigma2)};
    }

private:
    double l, sigma_f, sigma_n;
    std::vector<std::vector<int>> X;
    std::vector<double> y;
    std::vector<std::vector<double>> K, K_inv;

    // --- Math utilities ---

    double kernel(const std::vector<int>& a, const std::vector<int>& b) const {
        double sqdist = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = double(a[i]) - double(b[i]);
            sqdist += diff * diff;
        }
        return sigma_f * sigma_f * std::exp(-0.5 * sqdist / (l * l));
    }

    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

    static std::vector<double> matVecMul(const std::vector<std::vector<double>>& M,
                                         const std::vector<double>& v)
    {
        std::vector<double> result(M.size(), 0.0);
        for (size_t i = 0; i < M.size(); ++i)
            for (size_t j = 0; j < v.size(); ++j)
                result[i] += M[i][j] * v[j];
        return result;
    }

    // --- Simple matrix inversion using Gaussian elimination ---
    static std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>>& A) {
        const int n = (int)A.size();
        std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));

        // Form [A | I]
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                aug[i][j] = A[i][j];
            aug[i][n + i] = 1.0;
        }

        // Gaussian elimination
        for (int i = 0; i < n; ++i) {
            // Find pivot
            double pivot = aug[i][i];
            if (std::fabs(pivot) < 1e-12) {
                // Try to swap with a lower row
                for (int k = i + 1; k < n; ++k) {
                    if (std::fabs(aug[k][i]) > 1e-12) {
                        std::swap(aug[i], aug[k]);
                        pivot = aug[i][i];
                        break;
                    }
                }
            }
            if (std::fabs(pivot) < 1e-12)
                throw std::runtime_error("Matrix inversion failed: singular matrix");

            // Normalize pivot row
            for (int j = 0; j < 2 * n; ++j)
                aug[i][j] /= pivot;

            // Eliminate other rows
            for (int k = 0; k < n; ++k) {
                if (k == i) continue;
                double factor = aug[k][i];
                for (int j = 0; j < 2 * n; ++j)
                    aug[k][j] -= factor * aug[i][j];
            }
        }

        // Extract right half (the inverse)
        std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                inv[i][j] = aug[i][n + j];
        return inv;
    }
};
