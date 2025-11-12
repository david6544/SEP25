#if defined(BILINEAR) || defined(TESTING)
#ifndef BILINEAR_MODEL_H
#define BILINEAR_MODEL_H

/**
 * @file BilinearModel.hpp
 * @brief Bilinear interpolation model used for stochastic sampling and state-space mapping.
 *
 * @details
 * High-level overview:
 * The BilinearModel implements a sampling-and-interpolation strategy to map a
 * discrete state space (2D or 3D) into estimated probability values. It
 * performs stochastic sampling by issuing multiple queries per grid point,
 * aggregates samples, and uses bilinear (2D) or trilinear (3D) interpolation to
 * fill in values for unobserved coordinates. A KD-tree (KNNTree) is built from
 * sampled points to support nearest-neighbour queries and N-dimensional fallbacks.
 *
 * Relation to the state-space mapping challenge:
 * - The model trades off spatial coverage vs per-point accuracy by selecting
 *   a grid stride and samples-per-point. Coverage helps capture discontinuities
 *   in the function while repeated stochastic samples reduce variance at each
 *   location.
 * - For 3D spaces the Y-axis is treated as wrapping; the interpolation and
 *   nearest-neighbour logic account for this when computing distances and
 *   fractional interpolation coordinates.
 * - Confidence metrics (based on sample count and variance) are cached to
 *   prefer high-confidence estimates when available and fall back to KNN-based
 *   interpolation otherwise.
 *
 * This header documents the public contract: generate queries via
 * `get_next_query()`, record results with `update_prediction(...)`, and query
 * estimates with `get_value_at(...)`.
 *
 * @author David
 * @date 2025-11-02
 */

#include "Model.hpp"
#include "Tools/KNNAlgorithm.hpp"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * BilinearInterpolationModel
 * 
 * Implements stochastic sampling with bilinear interpolation.
 * Similar to the Python StochasticCheckerboardModel:
 * 1. Aggregates multiple stochastic queries per coordinate
 * 2. Computes average probability at sampled points
 * 3. Uses bilinear interpolation to fill in missing points
 */
class BilinearModel : public Model {
public:
    /**
     * @brief Construct a new BilinearModel
     *
     * @param dimensions number of dimensions (2 or 3)
     * @param dimensionSize size of each dimension
     * @param totalQueries total query budget for stochastic sampling
     */
    BilinearModel(int dimensions, int dimensionSize, int totalQueries);

    /**
     * @brief Destroy the BilinearModel
     */
    ~BilinearModel();

    /**
     * @brief Request the next stochastic query point.
     *
     * The model may return the same coordinate multiple times to collect
     * repeated stochastic samples at a location. Callers should pass the
     * observed value to `update_prediction`.
     *
     * @return std::vector<int> coordinate of the next query
     */
    std::vector<int> get_next_query() override;

    /**
     * @brief Record the result of a query for a coordinate.
     *
     * Aggregates samples for the given coordinate (sum, sum of squares, count)
     * and triggers KD-tree rebuild when sampling completes.
     *
     * @param query coordinate vector queried
     * @param result observed probability value in [0,1]
     */
    void update_prediction(const std::vector<int>& query, double result) override;

    /**
     * @brief Get the model's current estimate for a coordinate.
     *
     * Returns an exact averaged value if the coordinate has been sampled,
     * otherwise uses bilinear/trilinear interpolation or KD-tree based
     * interpolation as a fallback.
     *
     * @param query coordinate vector to estimate
     * @return double estimated probability in [0,1]
     */
    double get_value_at(const std::vector<int>& query) override;

private:
    struct GridPoint {
        std::vector<int> coords;
        double sum = 0.0;
        double sum_squares = 0.0;  // For variance calculation
        int count = 0;
        
        double average() const {
            return count > 0 ? sum / count : -1.0;  // -1 indicates no data
        }
        
        double variance() const {
            if (count < 2) return 0.0;
            double mean = sum / count;
            return (sum_squares / count) - (mean * mean);
        }
        
        double confidence() const {
            // Higher count and lower variance = higher confidence
            if (count == 0) return 0.0;
            double var = variance();
            // Confidence score: reduces with variance, increases with sample count
            return count / (1.0 + var);
        }
    };

    // Storage for sampled points
    std::unordered_map<int, GridPoint> sampledPoints;  // key = linearized index
    KNNTree* knnTree = nullptr;  // KD-tree for efficient nearest neighbor queries
    std::vector<std::pair<double, std::vector<int>>> treeData;  // Data for rebuilding tree
    std::unordered_map<int, double> confidenceCache;  // Cache confidence scores by index
    
    // Precomputed coordinate sets for fast trilinear interpolation
    std::vector<int> known_x, known_y, known_z;  // Sorted coordinate lists
    bool coordsCached = false;
    
    // Query generation
    int currentQueryPoint = 0;
    int repeatQueries = 0;
    int samplesPerPoint = 0;
    std::vector<int> currentPoint;
    
    // Grid sampling
    std::vector<std::vector<int>> gridPoints;
    int gridStride = 1;
    
    // Helper functions
    int coordsToIndex(const std::vector<int>& coords) const;
    std::vector<int> indexToCoords(int index) const;
    void generateGridPoints();
    void rebuildKDTree();  // Rebuild KD-tree after sampling phase
    double getNearestValue(const std::vector<int>& query) const;
    int getAdaptiveK(const std::vector<int>& query) const;  // Adaptive k based on local density
    
    // 2D-specific interpolation helpers
    double interpolate2D(int r, int c) const;
    void findSurroundingPoints2D(int r, int c, int& r_left, int& r_right, 
                                  int& c_left, int& c_right) const;
    
    // 3D trilinear interpolation
    double interpolate3D(int x, int y, int z) const;
    void findSurroundingPoints3D(int x, int y, int z,
                                  int& x0, int& x1,
                                  int& y0, int& y1,
                                  int& z0, int& z1) const;
    
    // N-D interpolation using KD-tree (for dimensions > 3)
    double interpolateND(const std::vector<int>& query) const;
};

#endif // BILINEAR_MODEL_H
#endif // BILINEAR || TESTING
