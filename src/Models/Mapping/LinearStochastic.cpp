#include "LinearStochastic.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <Eigen/Dense>
#include <iostream>

LinearStochastic::LinearStochastic(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power, double offset) : Mapping(data) {
    this->queriedPoints = data;
    this->knnTree = new KNNTree(this->queriedPoints);
    this->maxNeighbours = maxNeighbours;
    std::cout << maxNeighbours << std::endl;
    this->power = power;
    this->offset = offset;
}

LinearStochastic::~LinearStochastic(){
    delete this->knnTree;
}

using Matrix = std::vector<std::vector<double>>;

Matrix kroneckerProduct(const Matrix &A, const Matrix &B) {
    size_t aRows = A.size(), aCols = A[0].size();
    size_t bRows = B.size(), bCols = B[0].size();

    Matrix result(aRows * bRows, std::vector<double>(aCols * bCols));
    for (size_t i = 0; i < aRows; i++) {
        for (size_t j = 0; j < aCols; j++) {
            for (size_t p = 0; p < bRows; p++) {
                for (size_t q = 0; q < bCols; q++) {
                    result[i * bRows + p][j * bCols + q] = A[i][j] * B[p][q];
                }
            }
        }
    }
    return result;
}

Matrix buildM1(double x1, double x2) {
    return {
        { x2, -1 },
        { -x1, 1 }
    };
}

double LinearStochastic::predict(std::vector<int> query) {
    int dims = query.size();

    // Step 1: Get nearest neighbours (we use them only to derive local bounds)
    auto neighbors = this->knnTree->getKNearest(query, this->maxNeighbours);
    if (neighbors.empty()) return this->offset;

    // Step 2: For each dimension, find min and max (two closest unique points)
    std::vector<std::pair<double, double>> bounds(dims);
    for (int d = 0; d < dims; ++d) {
        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::lowest();
        for (auto &n : neighbors) {
            double val = n.point[d];
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
        if (std::fabs(maxVal - minVal) < 1e-9) {
            // Degenerate case: bump slightly to avoid zero division
            maxVal = minVal + 1e-6;
        }
        bounds[d] = { minVal, maxVal };
    }

    // Step 3: Multilinear interpolation via corner weights
    int corners = 1 << dims; // 2^n corners
    double prediction = 0.0;
    double totalWeight = 0.0;

    // Convert query coords to double
    std::vector<double> qd(dims);
    for (int d = 0; d < dims; ++d) qd[d] = static_cast<double>(query[d]);

    for (int i = 0; i < corners; ++i) {
        // build corner coordinate and compute weight
        double weight = 1.0;
        std::vector<int> cornerInt(dims);
        for (int d = 0; d < dims; ++d) {
            bool high = (i >> d) & 1;
            double x1 = bounds[d].first;
            double x2 = bounds[d].second;
            double denom = (x2 - x1);
            // safety:
            if (std::fabs(denom) < 1e-12) denom = 1e-12;

            if (!high) {
                // corner uses x1
                cornerInt[d] = static_cast<int>(std::lround(x1));
                weight *= (x2 - qd[d]) / denom;
            } else {
                // corner uses x2
                cornerInt[d] = static_cast<int>(std::lround(x2));
                weight *= (qd[d] - x1) / denom;
            }
        }

        // get value at corner (nearest recorded data point)
        auto cornerNeighbors = this->knnTree->getKNearest(cornerInt, 1);
        double ycorner = cornerNeighbors.front().value;

        prediction += weight * ycorner;
        totalWeight += weight;
    }
    

    return prediction;
}
