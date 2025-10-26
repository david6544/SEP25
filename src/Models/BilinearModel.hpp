#if defined(BILINEAR) || defined(TESTING)
#ifndef BILINEAR_MODEL_H
#define BILINEAR_MODEL_H

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
    BilinearModel(int dimensions, int dimensionSize, int totalQueries);
    ~BilinearModel();

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int>& query, double result) override;
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
    
    // N-D interpolation using KD-tree (much faster!)
    double interpolateND(const std::vector<int>& query) const;
};

#endif // BILINEAR_MODEL_H
#endif // BILINEAR || TESTING
