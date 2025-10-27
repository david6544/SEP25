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
        // Strategy: For discontinuous/unknown data, prioritize spatial coverage
        // With 300k queries and 100^3 space, we need balance between:
        // - Enough samples per point for reliable stochastic estimates (15-20 minimum)
        // - Enough spatial coverage to capture discontinuities
        
        // Step 3: 34^3 ≈ 39k points × 7 samples - too few samples, high variance
        // Step 4: 25^3 = 15.6k points × 19 samples - good coverage, decent samples
        // Step 5: 20^3 = 8k points × 37 samples - less coverage, better samples
        
        // For potentially discontinuous data, use smaller step for better coverage
        int step = 4;  // 15,625 points × 19 samples = better spatial resolution
        
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
    double max_dist = 0.0;
    
    for (const auto& neighbor : neighbors) {
        double dist = neighbor.distance;
        double value = neighbor.value;
        
        // Exact match
        if (dist < 1e-6) {
            return value;
        }
        
        max_dist = std::max(max_dist, dist);
    }
    
    // If we're far from all samples, regress toward the global mean
    // to avoid over-confident extrapolation
    double global_mean = 0.3;  // Conservative estimate
    if (!neighbors.empty() && max_dist > 10.0) {
        // Calculate sample mean from neighbors
        double sample_mean = 0.0;
        for (const auto& neighbor : neighbors) {
            sample_mean += neighbor.value;
        }
        sample_mean /= neighbors.size();
        
        // Blend between sample mean and global mean based on distance
        double blend_factor = std::min(1.0, max_dist / 20.0);  // 0 at dist=0, 1 at dist=20+
        return sample_mean * (1.0 - blend_factor) + global_mean * blend_factor;
    }
    
    for (const auto& neighbor : neighbors) {
        double dist = neighbor.distance;
        double value = neighbor.value;
        
        // Get cached confidence score for this neighbor
        int idx = coordsToIndex(neighbor.point);
        double confidence = 1.0;  // Default confidence
        
        auto conf_it = confidenceCache.find(idx);
        if (conf_it != confidenceCache.end()) {
            confidence = conf_it->second;
        }
        
        // Inverse distance weighting (not squared - less aggressive)
        // This provides smoother interpolation and reduces overfitting to nearest point
        // Normalize confidence to range [0.7, 1.0] to avoid over-weighting
        double normalized_confidence = 0.7 + 0.3 * (confidence / std::max(1.0, confidence));
        double dist_weight = 1.0 / (dist + 1.0);  // Add 1 to avoid division issues
        double total_weight = dist_weight * normalized_confidence;
        
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
    
    // For regular grid sampling with step=4, we want k=27 (3x3x3 neighborhood)
    // This captures the local cube around the query point
    // Using fewer neighbors leads to nearest-neighbor behavior (not interpolation)
    // Using many more neighbors over-smooths and loses local structure
    
    // For 3D with regular grid:
    // - 8 neighbors = corners of cube (trilinear interpolation)
    // - 27 neighbors = 3x3x3 cube (better for irregular or missing data)
    // - 64 neighbors = 4x4x4 cube (smooths too much)
    
    int k = (dimensions == 2) ? 9 : 27;  // 3x3 for 2D, 3x3x3 for 3D
    
    // Ensure k doesn't exceed available data
    return std::min(k, (int)treeData.size());
}

#endif // BILINEAR || TESTING
