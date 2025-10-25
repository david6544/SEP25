#if defined(KALMAN) || defined(TESTING)
#include "KalmanFilter.hpp"
#include <iostream>
#include <random>

KalmanFilter::KalmanFilter(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries),
      state(Eigen::VectorXd::Zero(dimensions)),
      covariance(Eigen::MatrixXd::Identity(dimensions, dimensions)),
      processNoise(Eigen::MatrixXd::Identity(dimensions, dimensions) * 0.1),
      measurementNoise(Eigen::MatrixXd::Identity(dimensions, dimensions) * 0.1),
      transitionMatrix(Eigen::MatrixXd::Identity(dimensions, dimensions)),
      measurementMatrix(Eigen::MatrixXd::Identity(dimensions, dimensions)) {
    // Initialization of the Kalman filter
}

std::vector<int> KalmanFilter::get_next_query() {
    // Predict the next state
    predict();

    // Convert the predicted state to a query vector
    std::vector<int> query(state.size());
    for (int i = 0; i < state.size(); ++i) {
        query[i] = static_cast<int>(state[i]);
    }
    return query;
}

void KalmanFilter::update_prediction(const std::vector<int> &query, double result) {
    // Convert query to Eigen vector
    Eigen::VectorXd measurement(query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        measurement[i] = query[i];
    }

    // Update the state with the measurement
    update(measurement);
}

double KalmanFilter::get_value_at(const std::vector<int> &query) {
    // Retrieve the value at the given query (state estimate)
    if (query.size() != state.size()) {
        throw std::invalid_argument("Query size does not match state dimensions.");
    }
    // Create a temporary VectorXd and copy values
    Eigen::VectorXd query_vec(query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        query_vec[i] = static_cast<double>(query[i]);
    }
    return state.dot(query_vec);
}

void KalmanFilter::predict() {
    // Predict the next state
    state = transitionMatrix * state;
    covariance = transitionMatrix * covariance * transitionMatrix.transpose() + processNoise;
}

void KalmanFilter::update(const Eigen::VectorXd &measurement) {
    // Compute Kalman gain
    Eigen::MatrixXd S = measurementMatrix * covariance * measurementMatrix.transpose() + measurementNoise;
    Eigen::MatrixXd K = covariance * measurementMatrix.transpose() * S.inverse();

    // Update state and covariance
    state = state + K * (measurement - measurementMatrix * state);
    covariance = (Eigen::MatrixXd::Identity(state.size(), state.size()) - K * measurementMatrix) * covariance;
}

#endif // KALMAN || TESTING