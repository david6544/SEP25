#if defined(BILINEAR) || defined(TESTING)
#include "BilinearModel.hpp"
#include <cmath>
#include <algorithm>
#include <set>

BilinearModel::BilinearModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) {
    
    // Generate grid points first
    generateGridPoints();
    
    // Calculate samples per point based on actual grid size
    // Distribute queries evenly across all grid points
    samplesPerPoint = std::max(1, totalQueries / std::max(1, (int)gridPoints.size()));
    
    // Reserve space for tree data
    treeData.reserve(gridPoints.size());
}

BilinearModel::~BilinearModel() {
    if (knnTree) {
        delete knnTree;
    }
}

void BilinearModel::generateGridPoints() {
    if (dimensions == 2) {
        // 2D grid: use regular sampling
        int step = std::max(2, dimensionSize / 10);
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
        // 3D space: maximize grid coverage with available budget
        // Strategy: balance between coverage (spatial resolution) and sample quality
        // For stochastic binary queries, need enough samples per point for reliable estimates
        
        // With 300k queries and 100^3 space:
        // Option 1: ~10k points × 30 samples = better estimates, step=5
        // Option 2: ~15k points × 20 samples = better coverage, step=4
        // Option 3: ~6k points × 50 samples = best estimates, step=6
        
        // Choose based on expected smoothness of the function
        // For potentially discontinuous data, prioritize coverage (smaller step)
        int step = 5;  // Direct choice: 8000 points × 37.5 samples per point
        
        // Generate regular 3D grid
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
        
        // Also add boundary points to ensure we capture edges
        // Add corners of the cube
        for (int x : {0, dimensionSize-1}) {
            for (int y : {0, dimensionSize-1}) {
                for (int z : {0, dimensionSize-1}) {
                    gridPoints.push_back({x, y, z});
                }
            }
        }
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
    gp.sum_squares += result * result;  // Track sum of squares for variance
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
    
    // Build data vector for KD-tree and cache confidence scores
    treeData.clear();
    treeData.reserve(sampledPoints.size());
    confidenceCache.clear();
    
    for (const auto& kv : sampledPoints) {
        if (kv.second.count > 0) {
            treeData.push_back({kv.second.average(), kv.second.coords});
            confidenceCache[kv.first] = kv.second.confidence();  // Cache confidence by index
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
        return 0.3;  // Conservative default based on typical distributions
    }
    
    // Use adaptive k based on local density
    const int k = getAdaptiveK(query);
    std::vector<KNNTree::KDResult> neighbors = knnTree->getKNearest(query, k);
    
    if (neighbors.empty()) {
        return 0.3;  // Conservative default
    }
    
    // Confidence-weighted IDW interpolation
    double sum_weights = 0.0;
    double sum_values = 0.0;
    
    for (const auto& neighbor : neighbors) {
        double dist = neighbor.distance;
        double value = neighbor.value;
        
        // Exact match
        if (dist < 1e-6) {
            return value;
        }
        
        // Get cached confidence score for this neighbor
        int idx = coordsToIndex(neighbor.point);
        double confidence = 1.0;  // Default confidence
        
        auto conf_it = confidenceCache.find(idx);
        if (conf_it != confidenceCache.end()) {
            confidence = conf_it->second;
        }
        
        // Inverse squared distance weighting, modulated by confidence
        // Higher confidence = more weight, lower confidence = less weight
        double dist_weight = 1.0 / (dist * dist);
        double total_weight = dist_weight * (0.5 + 0.5 * confidence);  // confidence factor: 0.5 to 1.0
        
        sum_weights += total_weight;
        sum_values += total_weight * value;
    }
    
    return sum_weights > 0 ? sum_values / sum_weights : 0.3;  // Conservative default
}

double BilinearModel::getNearestValue(const std::vector<int>& query) const {
    if (sampledPoints.empty()) {
        return 0.3;
    }
    
    double minDist = std::numeric_limits<double>::infinity();
    double nearestValue = 0.3;
    
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

int BilinearModel::getAdaptiveK(const std::vector<int>& query) const {
    if (!knnTree || treeData.empty()) {
        return 8;  // Default k
    }
    
    // Sample a few nearest neighbors to estimate local density
    const int sample_k = std::min(4, (int)treeData.size());
    std::vector<KNNTree::KDResult> sample = knnTree->getKNearest(query, sample_k);
    
    if (sample.empty()) return 8;
    
    // Calculate average distance to nearest neighbors
    double avg_dist = 0.0;
    for (const auto& neighbor : sample) {
        avg_dist += neighbor.distance;
    }
    avg_dist /= sample.size();
    
    // Adaptive k based on local density:
    // - Sparse regions (large avg_dist): use more neighbors (up to 32)
    // - Dense regions (small avg_dist): use fewer neighbors (down to 4)
    
    // Normalize distance by dimension size
    double normalized_dist = avg_dist / (dimensionSize * std::sqrt(dimensions));
    
    int adaptive_k;
    if (normalized_dist > 0.3) {
        // Sparse region - use many neighbors
        adaptive_k = 32;
    } else if (normalized_dist > 0.15) {
        // Medium density
        adaptive_k = 16;
    } else if (normalized_dist > 0.05) {
        // Dense region
        adaptive_k = 8;
    } else {
        // Very dense region
        adaptive_k = 4;
    }
    
    // Ensure k doesn't exceed available data
    return std::min(adaptive_k, (int)treeData.size());
}

#endif // BILINEAR || TESTING
