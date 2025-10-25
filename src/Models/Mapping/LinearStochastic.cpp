#include "LinearStochastic.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <Eigen/Dense>

LinearStochastic::LinearStochastic(std::vector<std::pair<double, std::vector<int>>>& data, int maxNeighbours, int power, double offset) : Mapping(data) {
    this->queriedPoints = data;
    this->knnTree = new KNNTree(this->queriedPoints);
    this->maxNeighbours = maxNeighbours;
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

    // Step 1: Get nearest neighbours
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
            // Degenerate case: no variation
            bounds[d] = {minVal, minVal + 1.0};
        } else {
            bounds[d] = {minVal, maxVal};
        }
    }

    // Step 3: Build M_n = ⊗ buildM1(x1, x2)
    Matrix M = buildM1(bounds[0].first, bounds[0].second);
    for (int d = 1; d < dims; ++d) {
        M = kroneckerProduct(M, buildM1(bounds[d].first, bounds[d].second));
    }

    // Step 4: Build y vector (2^n corner values)
    int corners = std::pow(2, dims);
    Eigen::VectorXd y(corners);
    for (int i = 0; i < corners; ++i) {
        std::vector<int> corner = query;
        for (int d = 0; d < dims; ++d) {
            bool high = (i >> d) & 1;
            corner[d] = high ? (int)bounds[d].second : (int)bounds[d].first;
        }
        // Find closest neighbor for this corner
        auto cornerNeighbors = this->knnTree->getKNearest(corner, 1);
        y(i) = cornerNeighbors.front().value;
    }

    // Step 5: Solve M * a = y
    Eigen::MatrixXd M_eig(M.size(), M[0].size());
    for (size_t i = 0; i < M.size(); ++i)
        for (size_t j = 0; j < M[i].size(); ++j)
            M_eig(i, j) = M[i][j];

    Eigen::VectorXd a = M_eig.colPivHouseholderQr().solve(y);

    // Step 6: Build query basis b(q)
    Matrix b = {{1, static_cast<double>(query[0])}};
    for (int d = 1; d < dims; ++d)
        b = kroneckerProduct(b, {{1, static_cast<double>(query[d])}});

    // Step 7: Compute prediction = b(q)^T * a
    Eigen::RowVectorXd bVec(b[0].size());
    for (size_t i = 0; i < b[0].size(); ++i)
        bVec(i) = b[0][i];

    double prediction = (bVec * a)(0);

    return prediction + this->offset;
}
