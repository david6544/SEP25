#ifndef KALMANFILTER_HPP
#define KALMANFILTER_HPP

#include "Model.hpp"
#include <Eigen/Dense>
#include <vector>

class KalmanFilter : public Model {
private:
    // State tracking
    Eigen::VectorXd state;
    Eigen::MatrixXd covariance;
    Eigen::MatrixXd processNoise;
    Eigen::MatrixXd measurementNoise;
    Eigen::MatrixXd transitionMatrix;
    Eigen::MatrixXd measurementMatrix;
    
    // Observation storage for interpolation
    struct Observation {
        std::vector<int> position;
        double value;
    };
    std::vector<Observation> observations;
    
    void predict();
    void update(const Eigen::VectorXd &measurement);

public:
    KalmanFilter(int dimensions, int dimensionSize, int totalQueries);
    
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;
};

#endif // KALMANFILTER_HPP