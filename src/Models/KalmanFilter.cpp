#if defined(KALMAN) || defined(TESTING)
#include "KalmanFilter.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

KalmanFilter::KalmanFilter(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries),
      state(Eigen::VectorXd::Zero(dimensions)),
      covariance(Eigen::MatrixXd::Identity(dimensions, dimensions) * 100.0),
      processNoise(Eigen::MatrixXd::Identity(dimensions, dimensions) * 0.1),
      measurementNoise(Eigen::MatrixXd::Identity(dimensions, dimensions) * 1.0),
      transitionMatrix(Eigen::MatrixXd::Identity(dimensions, dimensions)),
      measurementMatrix(Eigen::MatrixXd::Identity(dimensions, dimensions)) {
   
    // Initialize state with random values within dimension size
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, dimensionSize);
   
    // Initialize state with random values
    for (int i = 0; i < dimensions; ++i) {
        state[i] = dis(gen);
    }
    
    // Initialize observation storage
    observations.reserve(totalQueries);
}

std::vector<int> KalmanFilter::get_next_query() {
    std::vector<int> query(dimensions);
    
    // Strategy: Balance exploration and exploitation
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> exploration_prob(0.0, 1.0);
    
    // Exploration vs exploitation trade-off
    double exploration_rate = std::max(0.1, 1.0 - (double)observations.size() / totalQueries);
    
    if (exploration_prob(gen) < exploration_rate || observations.empty()) {
        // EXPLORATION: Random sampling with focus on unexplored regions
        std::uniform_int_distribution<> pos_dist(0, dimensionSize - 1);
        for (int i = 0; i < dimensions; ++i) {
            query[i] = pos_dist(gen);
        }
    } else {
        // EXPLOITATION: Query near best observed locations
        if (!observations.empty()) {
            // Find the best observation so far
            auto best_obs = std::min_element(observations.begin(), observations.end(),
                [](const Observation& a, const Observation& b) {
                    return a.value < b.value;
                });
            
            // Sample around the best point with Gaussian noise
            std::normal_distribution<> noise(0.0, dimensionSize * 0.2);
            
            for (int i = 0; i < dimensions; ++i) {
                double noisy_pos = best_obs->position[i] + noise(gen);
                query[i] = std::max(0, std::min(dimensionSize - 1, 
                                                 static_cast<int>(std::round(noisy_pos))));
            }
        } else {
            // Fallback to center
            for (int i = 0; i < dimensions; ++i) {
                query[i] = dimensionSize / 2;
            }
        }
    }
    
    return query;
}

void KalmanFilter::update_prediction(const std::vector<int> &query, double result) {
    if (query.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Query dimension size is out of range");
    }
    
    // Store observation
    Observation obs;
    obs.position = query;
    obs.value = result;
    observations.push_back(obs);
    
    // Update state to move towards better regions
    // Use weighted average of observations, giving more weight to better values
    if (observations.size() >= 3) {
        // Find top 30% best observations
        size_t top_k = std::max(size_t(1), observations.size() * 3 / 10);
        std::vector<Observation> sorted_obs = observations;
        std::partial_sort(sorted_obs.begin(), 
                         sorted_obs.begin() + top_k, 
                         sorted_obs.end(),
                         [](const Observation& a, const Observation& b) {
                             return a.value < b.value;
                         });
        
        // Update state as weighted centroid of best observations
        Eigen::VectorXd new_state = Eigen::VectorXd::Zero(dimensions);
        double total_weight = 0.0;
        
        for (size_t i = 0; i < top_k; ++i) {
            // Weight by inverse rank (best gets highest weight)
            double weight = 1.0 / (i + 1.0);
            total_weight += weight;
            
            for (int d = 0; d < dimensions; ++d) {
                new_state[d] += weight * sorted_obs[i].position[d];
            }
        }
        
        new_state /= total_weight;
        
        // Smooth transition to new state
        double alpha = 0.3;  // Learning rate
        state = alpha * new_state + (1.0 - alpha) * state;
    }
}

double KalmanFilter::get_value_at(const std::vector<int> &query) {
    if (query.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Query dimension size is out of range");
    }
    
    if (observations.empty()) {
        return 0.0;  // No data yet
    }
    
    // Use Inverse Distance Weighting (IDW) for interpolation
    double sum_weighted_values = 0.0;
    double sum_weights = 0.0;
    const double epsilon = 1e-10;  // Avoid division by zero
    const double power = 2.0;  // IDW power parameter
    
    // Find k nearest neighbors
    size_t k = std::min(size_t(10), observations.size());
    std::vector<std::pair<double, double>> distances_and_values;
    
    for (const auto& obs : observations) {
        double dist_sq = 0.0;
        for (size_t i = 0; i < query.size(); ++i) {
            double diff = query[i] - obs.position[i];
            dist_sq += diff * diff;
        }
        double dist = std::sqrt(dist_sq);
        distances_and_values.push_back({dist, obs.value});
    }
    
    // Sort by distance
    std::partial_sort(distances_and_values.begin(),
                     distances_and_values.begin() + k,
                     distances_and_values.end());
    
    // Check for exact match
    if (distances_and_values[0].first < epsilon) {
        return distances_and_values[0].second;
    }
    
    // IDW interpolation using k nearest neighbors
    for (size_t i = 0; i < k; ++i) {
        double dist = distances_and_values[i].first;
        double value = distances_and_values[i].second;
        double weight = 1.0 / (std::pow(dist, power) + epsilon);
        
        sum_weighted_values += weight * value;
        sum_weights += weight;
    }
    
    return sum_weighted_values / sum_weights;
}

void KalmanFilter::predict() {
    // Not used in this implementation - kept for interface compatibility
}

void KalmanFilter::update(const Eigen::VectorXd &measurement) {
    // Not used in this implementation - kept for interface compatibility
}

#endif // KALMAN || TESTING