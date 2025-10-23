#if defined(GEK) || defined(TESTING)
#ifndef GEK_MAPPING_H
#define GEK_MAPPING_H

#include "Mapping.hpp"
#include <vector>
#include <Eigen/Dense>

/**
 * @brief GEKMapping implements Gradient-Enhanced Kriging interpolation.
 * 
 * This is a form of Gaussian process regression that can use local neighborhoods
 * for efficiency and supports periodic (cylindrical) boundary conditions.
 */
class GEKMapping : public Mapping {
public:
    /**
     * @brief Construct a new GEKMapping object
     * 
     * @param data Vector of (value, coordinates) pairs from queried points
     * @param dimensions Number of dimensions in the space
     * @param dimensionSize Size of each dimension
     * @param use_local_neighborhood Whether to use local neighborhoods for prediction
     * @param local_k Number of nearest neighbors to use for local solves
     */
    GEKMapping(std::vector<std::pair<double, std::vector<int>>>& data,
               int dimensions,
               int dimensionSize,
               bool use_local_neighborhood = true,
               int local_k = 64);
    
    ~GEKMapping() override;
    
    /**
     * @brief Predict the value at a given query point using GEK interpolation
     * 
     * @param query The coordinates to predict at
     * @return double The predicted value
     */
    double predict(std::vector<int> query) override;
    
    /**
     * @brief Configure which dimensions have periodic boundary conditions
     * 
     * @param periodic Vector of booleans indicating which dimensions are periodic
     */
    void set_periodic_dimensions(const std::vector<bool>& periodic);
    
    /**
     * @brief Get the variance/uncertainty at a query point
     * 
     * @param query The coordinates to evaluate uncertainty at
     * @return double The predicted variance
     */
    double get_variance(const std::vector<int>& query);

private:
    int dimensions;
    int dimensionSize;
    double theta = 0.01;  // covariance length scale parameter
    bool use_local_neighborhood;
    int local_k;
    std::vector<bool> periodic_dims;
    
    // Global covariance matrix inverse (for non-local mode)
    Eigen::MatrixXd C_inv;
    bool trained = false;
    
    /**
     * @brief Compute covariance between two points
     */
    double covariance(const std::vector<int>& x, const std::vector<int>& y);
    
    /**
     * @brief Compute squared distance with periodic boundary handling
     */
    double compute_periodic_distance_squared(const std::vector<int>& x, const std::vector<int>& y);
    
    /**
     * @brief Train the global covariance matrix (if not using local neighborhoods)
     */
    void train();
    
    /**
     * @brief Select k nearest observed points to a query
     */
    std::vector<int> select_k_nearest_indices(const std::vector<int>& query, int k);
    
    /**
     * @brief Build local covariance matrix and target vector
     */
    void build_local_cov_and_targets(const std::vector<int>& indices,
                                     Eigen::MatrixXd& C_loc,
                                     Eigen::VectorXd& y_loc);
};

#endif // GEK_MAPPING_H
#endif // GEK || TESTING
