#if defined(KALMAN) || defined(TESTING)
#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include "Model.hpp"
#include <vector>
#include <Eigen/Dense> // For matrix operations

/**
 * @brief The KalmanFilter class implements a Kalman filter for state estimation.
 */
class KalmanFilter : public Model {
    Eigen::VectorXd state;         // State vector
    Eigen::MatrixXd covariance;    // Covariance matrix
    Eigen::MatrixXd processNoise;  // Process noise covariance
    Eigen::MatrixXd measurementNoise; // Measurement noise covariance
    Eigen::MatrixXd transitionMatrix; // State transition matrix
    Eigen::MatrixXd measurementMatrix; // Measurement matrix

public:
    KalmanFilter(int dimensions, int dimensionSize, int totalQueries);

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;

private:
    void predict();
    void update(const Eigen::VectorXd &measurement);
};

#endif // KALMAN_FILTER_H
#endif // KALMAN || TESTING