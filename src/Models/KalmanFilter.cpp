#if defined(KALMAN) || defined(TESTING)
#include "KalmanFilter.hpp"
#include <iostream>
#include <random>
#include <cmath>

KalmanFilter::KalmanFilter(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries),
      state(Eigen::VectorXd::Zero(dimensions + 1)),  // +1 for value component
      covariance(Eigen::MatrixXd::Identity(dimensions + 1, dimensions + 1) * 100.0),
      processNoise(Eigen::MatrixXd::Identity(dimensions + 1, dimensions + 1) * 0.1),
      measurementNoise(Eigen::MatrixXd::Identity(dimensions + 1, dimensions + 1) * 1.0),
      transitionMatrix(Eigen::MatrixXd::Identity(dimensions + 1, dimensions + 1)),
      measurementMatrix(Eigen::MatrixXd::Identity(dimensions + 1, dimensions + 1)) {
   
    // Initialize state with random values within dimension size
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, dimensionSize);
   
    // Initialize position components with random values (leave value component at 0)
    for (int i = 0; i < dimensions; ++i) {
        state[i] = dis(gen);
    }
    // Last component (value) remains 0
}

std::vector<int> KalmanFilter::get_next_query() {
    // Predict the next state
    predict();
    
    // Convert the predicted state to a query vector (excluding the value component)
    std::vector<int> query(get_dimensions());  // Use get_dimensions() instead
    for (int i = 0; i < get_dimensions(); ++i) {
        // Ensure query stays within bounds
        int pos = std::max(0, std::min(get_dimensionSize() - 1, static_cast<int>(state[i])));
        query[i] = pos;
    }
    return query;
}

void KalmanFilter::update_prediction(const std::vector<int> &query, double result) {
    // Verify query size matches dimensions
    if (query.size() != get_dimensions()) {
        throw std::invalid_argument("Query dimension size is out of range");
    }
    
    // Create augmented measurement vector that includes both position and result
    Eigen::VectorXd measurement(state.size());
    for (size_t i = 0; i < query.size(); ++i) {
        measurement[i] = query[i];
    }
    measurement[query.size()] = result;  // Add the function value to the measurement
   
    // Adaptive measurement noise based on the observation
    double pos_noise = 0.1;  // Position measurement noise
    double val_noise = std::max(0.1, std::abs(result) * 0.05);  // Value measurement noise
   
    measurementNoise = Eigen::MatrixXd::Identity(state.size(), state.size());
    for (size_t i = 0; i < query.size(); ++i) {
        measurementNoise(i, i) = pos_noise;
    }
    measurementNoise(query.size(), query.size()) = val_noise;
    
    // Update the state with the measurement
    update(measurement);
}

double KalmanFilter::get_value_at(const std::vector<int> &query) {
    if (query.size() != get_dimensions()) {
        throw std::invalid_argument("Query dimension size is out of range");
    }
    
    // The last component of the state vector represents our estimate of the function value
    return state[get_dimensions()];
}

void KalmanFilter::predict() {
    // Add small random walk to the state for exploration
    Eigen::VectorXd noise = Eigen::VectorXd::Random(state.size());
    state = transitionMatrix * state + noise * 0.1 * get_dimensionSize();
   
    // Update covariance
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