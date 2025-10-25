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

    // Step 2: Construct design matrix X and target vector y for WLS
    Eigen::MatrixXd X(neighbors.size(), dims + 1); // last column = bias term
    Eigen::VectorXd y(neighbors.size());
    Eigen::VectorXd w(neighbors.size());

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &pt = neighbors[i].point;
        double distance = 0.0;
        for (int d = 0; d < dims; ++d) distance += std::pow(query[d] - pt[d], 2);
        distance = std::sqrt(distance);
        double weight = 1.0 / (distance + 1e-6); // inverse-distance weighting
        w(i) = weight;

        for (int d = 0; d < dims; ++d) X(i, d) = pt[d];
        X(i, dims) = 1.0; // bias term
        y(i) = neighbors[i].value;
    }

    // Step 3: Apply weighted least squares: beta = (X^T W X)^-1 X^T W y
    Eigen::MatrixXd W = w.asDiagonal();
    Eigen::VectorXd beta;
    try {
        beta = (X.transpose() * W * X).ldlt().solve(X.transpose() * W * y);
    } catch (...) {
        // fallback to simple weighted average if inversion fails
        double pred = 0.0, totalW = 0.0;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            pred += w(i) * y(i);
            totalW += w(i);
        }
        return totalW > 1e-12 ? pred / totalW : this->offset;
    }

    // Step 4: Predict at the query point
    Eigen::VectorXd q(dims + 1);
    for (int d = 0; d < dims; ++d) q(d) = query[d];
    q(dims) = 1.0; // bias
    double prediction = beta.dot(q);

    return prediction;
}
