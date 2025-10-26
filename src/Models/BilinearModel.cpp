#if defined(BILINEAR) || defined(TESTING)
#include "BilinearModel.hpp"
#include <cmath>
#include <algorithm>
#include <set>

BilinearModel::BilinearModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) {
    
    // Determine samples per point based on total budget
    // Similar to stochastic model: use multiple samples per grid point
    int totalGridPoints = (int)std::pow(dimensionSize / 3.0, dimensions);  // Sparse sampling
    samplesPerPoint = std::max(3, totalQueries / std::max(1, totalGridPoints));
    
    // Generate sparse grid for sampling
    generateGridPoints();
    
    // Reserve space for tree data
    treeData.reserve(gridPoints.size());
}

BilinearModel::~BilinearModel() {
    if (knnTree) {
        delete knnTree;
    }
}

void BilinearModel::generateGridPoints() {
    // Create a sparse grid based on available query budget
    gridStride = std::max(2, (int)std::pow(dimensionSize, dimensions) / (totalQueries / samplesPerPoint));
    
    if (dimensions == 2) {
        // 2D grid
        int step = std::max(2, dimensionSize / 10);  // Sample ~100 points for a 100x100 grid
        for (int r = 0; r < dimensionSize; r += step) {
            for (int c = 0; c < dimensionSize; c += step) {
                gridPoints.push_back({r, c});
            }
        }
        // Ensure corners are included
        gridPoints.push_back({0, 0});
        gridPoints.push_back({0, dimensionSize-1});
        gridPoints.push_back({dimensionSize-1, 0});
        gridPoints.push_back({dimensionSize-1, dimensionSize-1});
    } else {
        // N-D: sample along main diagonal and some random points
        int numSamples = std::min(100, totalQueries / samplesPerPoint);
        int step = std::max(1, dimensionSize / (int)std::cbrt(numSamples));
        
        // Regular grid samples
        std::vector<int> point(dimensions);
        std::function<void(int)> generateRecursive = [&](int dim) {
            if (dim == dimensions) {
                gridPoints.push_back(point);
                return;
            }
            for (int i = 0; i < dimensionSize; i += step) {
                point[dim] = i;
                generateRecursive(dim + 1);
            }
        };
        generateRecursive(0);
    }
    
    // Remove duplicates
    std::set<int> seen;
    std::vector<std::vector<int>> unique;
    for (const auto& pt : gridPoints) {
        int idx = coordsToIndex(pt);
        if (seen.insert(idx).second) {
            unique.push_back(pt);
        }
    }
    gridPoints = unique;
}

int BilinearModel::coordsToIndex(const std::vector<int>& coords) const {
    int index = 0;
    int multiplier = 1;
    for (int i = dimensions - 1; i >= 0; --i) {
        index += coords[i] * multiplier;
        multiplier *= dimensionSize;
    }
    return index;
}

std::vector<int> BilinearModel::indexToCoords(int index) const {
    std::vector<int> coords(dimensions);
    for (int i = dimensions - 1; i >= 0; --i) {
        coords[i] = index % dimensionSize;
        index /= dimensionSize;
    }
    return coords;
}

std::vector<int> BilinearModel::get_next_query() {
    currentQuery++;
    
    // Check if we need to move to next grid point
    if (repeatQueries == 0) {
        if (currentQueryPoint >= (int)gridPoints.size()) {
            // Ran out of grid points, return center
            return std::vector<int>(dimensions, dimensionSize / 2);
        }
        currentPoint = gridPoints[currentQueryPoint];
        currentQueryPoint++;
        repeatQueries = samplesPerPoint;
    }
    
    repeatQueries--;
    return currentPoint;  // Return same point for stochastic sampling
}

void BilinearModel::update_prediction(const std::vector<int>& query, double result) {
    int idx = coordsToIndex(query);
    
    auto& gp = sampledPoints[idx];
    if (gp.coords.empty()) {
        gp.coords = query;
    }
    gp.sum += result;
    gp.count++;
    
    // Build KD-tree after all sampling is complete (when we've sampled all grid points)
    if (currentQuery == totalQueries && !knnTree) {
        rebuildKDTree();
    }
}

double BilinearModel::get_value_at(const std::vector<int>& query) {
    int idx = coordsToIndex(query);
    
    // Check if we have data for this exact point
    auto it = sampledPoints.find(idx);
    if (it != sampledPoints.end() && it->second.count > 0) {
        return it->second.average();
    }
    
    // Build KD-tree if not already built (lazy initialization for predictions)
    if (!knnTree && !sampledPoints.empty()) {
        const_cast<BilinearModel*>(this)->rebuildKDTree();
    }
    
    // Use interpolation
    if (dimensions == 2) {
        return interpolate2D(query[0], query[1]);
    } else {
        return interpolateND(query);
    }
}

void BilinearModel::rebuildKDTree() {
    // Clear old tree
    if (knnTree) {
        delete knnTree;
        knnTree = nullptr;
    }
    
    // Build data vector for KD-tree
    treeData.clear();
    treeData.reserve(sampledPoints.size());
    
    for (const auto& kv : sampledPoints) {
        if (kv.second.count > 0) {
            treeData.push_back({kv.second.average(), kv.second.coords});
        }
    }
    
    // Build KD-tree
    if (!treeData.empty()) {
        knnTree = new KNNTree(treeData);
    }
}

void BilinearModel::findSurroundingPoints2D(int r, int c, int& r_left, int& r_right, 
                                             int& c_left, int& c_right) const {
    // Find known rows and columns
    std::set<int> known_rows, known_cols;
    for (const auto& kv : sampledPoints) {
        if (kv.second.count > 0) {
            known_rows.insert(kv.second.coords[0]);
            known_cols.insert(kv.second.coords[1]);
        }
    }
    
    // Find surrounding coordinates
    r_left = r_right = -1;
    c_left = c_right = -1;
    
    for (int kr : known_rows) {
        if (kr <= r && (r_left == -1 || kr > r_left)) r_left = kr;
        if (kr >= r && (r_right == -1 || kr < r_right)) r_right = kr;
    }
    
    for (int kc : known_cols) {
        if (kc <= c && (c_left == -1 || kc > c_left)) c_left = kc;
        if (kc >= c && (c_right == -1 || kc < c_right)) c_right = kc;
    }
    
    // Handle edges
    if (r_left == -1) r_left = r_right;
    if (r_right == -1) r_right = r_left;
    if (c_left == -1) c_left = c_right;
    if (c_right == -1) c_right = c_left;
}

double BilinearModel::interpolate2D(int r, int c) const {
    if (sampledPoints.empty()) {
        return 0.5;  // Default fallback
    }
    
    int r_left, r_right, c_left, c_right;
    findSurroundingPoints2D(r, c, r_left, r_right, c_left, c_right);
    
    // If no surrounding points found, use nearest
    if (r_left == -1 || c_left == -1) {
        return getNearestValue({r, c});
    }
    
    // Get corner values
    std::vector<double> corner_values;
    std::vector<std::vector<int>> corners = {
        {r_left, c_left},
        {r_left, c_right},
        {r_right, c_left},
        {r_right, c_right}
    };
    
    for (const auto& corner : corners) {
        int idx = coordsToIndex(corner);
        auto it = sampledPoints.find(idx);
        if (it != sampledPoints.end() && it->second.count > 0) {
            corner_values.push_back(it->second.average());
        } else {
            // Fallback to nearest
            corner_values.push_back(getNearestValue(corner));
        }
    }
    
    // Perform bilinear interpolation
    if (r_left == r_right && c_left == c_right) {
        return corner_values[0];
    } else if (r_left == r_right) {
        // Interpolate along c
        double frac_c = (double)(c - c_left) / (c_right - c_left);
        return corner_values[0] * (1.0 - frac_c) + corner_values[1] * frac_c;
    } else if (c_left == c_right) {
        // Interpolate along r
        double frac_r = (double)(r - r_left) / (r_right - r_left);
        return corner_values[0] * (1.0 - frac_r) + corner_values[2] * frac_r;
    } else {
        // Full bilinear interpolation
        double frac_r = (double)(r - r_left) / (r_right - r_left);
        double frac_c = (double)(c - c_left) / (c_right - c_left);
        
        double top = corner_values[0] * (1.0 - frac_c) + corner_values[1] * frac_c;
        double bottom = corner_values[2] * (1.0 - frac_c) + corner_values[3] * frac_c;
        
        return top * (1.0 - frac_r) + bottom * frac_r;
    }
}

double BilinearModel::interpolateND(const std::vector<int>& query) const {
    // Use KD-tree for efficient nearest neighbor lookup
    if (!knnTree) {
        return 0.5;  // Default value if tree not built yet
    }
    
    // Find k nearest neighbors using KD-tree (O(log N) instead of O(N))
    const int k = std::min(8, (int)treeData.size());  // Use 8 neighbors for better interpolation
    std::vector<KNNTree::KDResult> neighbors = knnTree->getKNearest(query, k);
    
    if (neighbors.empty()) {
        return 0.5;
    }
    
    // IDW interpolation with inverse squared distance weighting
    double sum_weights = 0.0;
    double sum_values = 0.0;
    
    for (const auto& neighbor : neighbors) {
        double dist = neighbor.distance;
        double value = neighbor.value;
        
        // Exact match
        if (dist < 1e-6) {
            return value;
        }
        
        // Inverse squared distance weighting
        double weight = 1.0 / (dist * dist);
        sum_weights += weight;
        sum_values += weight * value;
    }
    
    return sum_weights > 0 ? sum_values / sum_weights : 0.5;
}

double BilinearModel::getNearestValue(const std::vector<int>& query) const {
    if (sampledPoints.empty()) {
        return 0.5;
    }
    
    double minDist = std::numeric_limits<double>::infinity();
    double nearestValue = 0.5;
    
    for (const auto& kv : sampledPoints) {
        if (kv.second.count == 0) continue;
        
        double dist = 0.0;
        for (int d = 0; d < dimensions; ++d) {
            double diff = query[d] - kv.second.coords[d];
            dist += diff * diff;
        }
        
        if (dist < minDist) {
            minDist = dist;
            nearestValue = kv.second.average();
        }
    }
    
    return nearestValue;
}

#endif // BILINEAR || TESTING
