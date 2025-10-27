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
        // 3D space: adapt sampling strategy based on expected function characteristics
        // Key insight from visualizations: smooth continuous functions with large regions
        // 
        // For SMOOTH functions (large connected regions, curved boundaries):
        // - Prioritize SPATIAL COVERAGE over samples per point
        // - Interpolation works well when function is continuous
        // - Better to sample more locations with fewer samples each
        //
        // Step 2: 50×50×50 = 125k points × 2 samples
        //   + Excellent coverage (12.5%), max dist 3.5 units
        //   - Only 2 samples: high variance per point (SE=0.35)
        //   - But interpolation smooths this out!
        //
        // Step 3: 34×34×34 = 39k points × 7 samples  
        //   + Good coverage (3.9%), reasonable variance (SE=0.19)
        //   + Best balance for smooth functions
        //
        // Step 4: 25×25×25 = 15.6k points × 19 samples
        //   + Low variance per point (SE=0.11)
        //   - Poor coverage (1.6%), misses structure
        
        // For smooth continuous data AND squared error penalty: balance coverage vs accuracy
        // Squared error heavily penalizes outliers (error^2), so we need:
        // 1. Better per-point accuracy (more samples) to reduce variance
        // 2. Still maintain reasonable coverage for interpolation
        // 
        // Step 3: 39k points × 7 samples (SE=0.19) - high outlier risk
        // Step 4: 15.6k points × 19 samples (SE=0.11) - better accuracy, still reasonable coverage
        int step = 4;  // 15,625 points × 19 samples - better for squared error metric
        
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
        // Note: Y-axis wraps, so y=0 and y=99 are the same point (don't duplicate)
        for (int x : {0, dimensionSize-1}) {
            for (int y : {0}) {  // Only y=0 since y wraps (y=99 is same as y=0)
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
    
    // Precompute coordinate sets for fast trilinear interpolation
    std::set<int> x_set, y_set, z_set;
    
    for (const auto& kv : sampledPoints) {
        if (kv.second.count > 0) {
            treeData.push_back({kv.second.average(), kv.second.coords});
            confidenceCache[kv.first] = kv.second.confidence();  // Cache confidence by index
            
            // Build coordinate sets
            if (dimensions >= 1) x_set.insert(kv.second.coords[0]);
            if (dimensions >= 2) y_set.insert(kv.second.coords[1]);
            if (dimensions >= 3) z_set.insert(kv.second.coords[2]);
        }
    }
    
    // Convert sets to sorted vectors for O(log m) binary search
    known_x.assign(x_set.begin(), x_set.end());
    known_y.assign(y_set.begin(), y_set.end());
    known_z.assign(z_set.begin(), z_set.end());
    coordsCached = true;
    
    // Build KD-tree
    if (!treeData.empty()) {
        knnTree = new KNNTree(treeData);
    }
}

void BilinearModel::findSurroundingPoints2D(int r, int c, int& r_left, int& r_right, 
                                             int& c_left, int& c_right) const {
    // Use precomputed coordinate lists for O(log m) binary search (same as 3D)
    if (coordsCached && !known_x.empty() && !known_y.empty()) {
        // Binary search on precomputed sorted lists
        auto it_r = std::lower_bound(known_x.begin(), known_x.end(), r);
        if (it_r != known_x.end() && *it_r == r) {
            r_left = r_right = r;
        } else {
            r_right = (it_r != known_x.end()) ? *it_r : -1;
            r_left = (it_r != known_x.begin()) ? *(--it_r) : -1;
        }
        
        auto it_c = std::lower_bound(known_y.begin(), known_y.end(), c);
        if (it_c != known_y.end() && *it_c == c) {
            c_left = c_right = c;
        } else {
            c_right = (it_c != known_y.end()) ? *it_c : -1;
            c_left = (it_c != known_y.begin()) ? *(--it_c) : -1;
        }
    } else {
        // Fallback: build coordinate sets (only if not cached)
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
    double result;
    if (r_left == r_right && c_left == c_right) {
        result = corner_values[0];
    } else if (r_left == r_right) {
        // Interpolate along c
        double frac_c = (double)(c - c_left) / (c_right - c_left);
        result = corner_values[0] * (1.0 - frac_c) + corner_values[1] * frac_c;
    } else if (c_left == c_right) {
        // Interpolate along r
        double frac_r = (double)(r - r_left) / (r_right - r_left);
        result = corner_values[0] * (1.0 - frac_r) + corner_values[2] * frac_r;
    } else {
        // Full bilinear interpolation
        double frac_r = (double)(r - r_left) / (r_right - r_left);
        double frac_c = (double)(c - c_left) / (c_right - c_left);
        
        double top = corner_values[0] * (1.0 - frac_c) + corner_values[1] * frac_c;
        double bottom = corner_values[2] * (1.0 - frac_c) + corner_values[3] * frac_c;
        
        result = top * (1.0 - frac_r) + bottom * frac_r;
    }
    
    // Safety clamp
    return std::max(0.0, std::min(1.0, result));
}

double BilinearModel::interpolateND(const std::vector<int>& query) const {
    if (dimensions == 3) {
        return interpolate3D(query[0], query[1], query[2]);
    }
    
    // Fallback to KNN for higher dimensions (shouldn't happen in practice)
    if (!knnTree) {
        return 0.3;
    }
    
    const int k = 27;  // 3^3 cube
    std::vector<KNNTree::KDResult> neighbors = knnTree->getKNearest(query, k);
    
    if (neighbors.empty()) {
        return 0.3;
    }
    
    // Check for exact match
    for (const auto& neighbor : neighbors) {
        if (neighbor.distance < 1e-6) {
            return neighbor.value;
        }
    }
    
    // Gentle IDW interpolation
    double sum_weights = 0.0;
    double sum_values = 0.0;
    
    for (const auto& neighbor : neighbors) {
        double dist = neighbor.distance;
        double dist_weight = 1.0 / (1.0 + dist / 5.0);
        sum_weights += dist_weight;
        sum_values += dist_weight * neighbor.value;
    }
    
    return sum_weights > 0 ? sum_values / sum_weights : 0.3;
}

double BilinearModel::getNearestValue(const std::vector<int>& query) const {
    if (sampledPoints.empty()) {
        return 0.3;
    }
    
    // Use KD-tree for O(log n) nearest neighbor lookup
    // Note: KD-tree uses Euclidean distance, which doesn't account for Y wrapping
    // For better accuracy with wrapping, we'd need a custom distance metric
    // But for fallback cases (rare), standard KD-tree is acceptable
    if (knnTree && !treeData.empty()) {
        auto neighbors = knnTree->getKNearest(query, 1);
        if (!neighbors.empty()) {
            return neighbors[0].value;
        }
    }
    
    // Fallback to linear search with WRAPPED distance for Y-axis (3D only)
    double minDist = std::numeric_limits<double>::infinity();
    double nearestValue = 0.3;
    
    for (const auto& kv : sampledPoints) {
        if (kv.second.count == 0) continue;
        
        double dist = 0.0;
        for (int d = 0; d < dimensions; ++d) {
            double diff = query[d] - kv.second.coords[d];
            
            // Y-axis wraps in 3D (d=1)
            if (dimensions == 3 && d == 1) {
                double wrapped_diff = std::min(std::abs(diff), dimensionSize - std::abs(diff));
                dist += wrapped_diff * wrapped_diff;
            } else {
                dist += diff * diff;
            }
        }
        
        if (dist < minDist) {
            minDist = dist;
            nearestValue = kv.second.average();
        }
    }
    
    return nearestValue;
}

double BilinearModel::interpolate3D(int x, int y, int z) const {
    if (sampledPoints.empty()) {
        return 0.3;  // Default fallback
    }
    
    // Find the 8 surrounding grid points for trilinear interpolation
    int x0, x1, y0, y1, z0, z1;
    findSurroundingPoints3D(x, y, z, x0, x1, y0, y1, z0, z1);
    
    // If we don't have surrounding points, fall back to nearest neighbor
    if (x0 == -1 || y0 == -1 || z0 == -1) {
        return getNearestValue({x, y, z});
    }
    
    // Get the 8 corner values of the cube
    std::vector<double> corner_values(8);
    std::vector<double> corner_confidence(8);  // Track confidence for each corner
    std::vector<std::vector<int>> corners = {
        {x0, y0, z0}, {x1, y0, z0},  // bottom face
        {x0, y1, z0}, {x1, y1, z0},
        {x0, y0, z1}, {x1, y0, z1},  // top face
        {x0, y1, z1}, {x1, y1, z1}
    };
    
    bool all_corners_sampled = true;
    for (size_t i = 0; i < corners.size(); ++i) {
        int idx = coordsToIndex(corners[i]);
        auto it = sampledPoints.find(idx);
        if (it != sampledPoints.end() && it->second.count > 0) {
            corner_values[i] = it->second.average();
            corner_confidence[i] = it->second.count;  // More samples = higher confidence
        } else {
            // Corner missing - this is dangerous for squared error!
            all_corners_sampled = false;
            corner_values[i] = getNearestValue(corners[i]);
            corner_confidence[i] = 0.1;  // Low confidence for interpolated corners
        }
    }
    
    // If any corners are missing, blend with local average to reduce outlier risk
    if (!all_corners_sampled) {
        // Calculate confidence-weighted average of available corners
        double sum_weighted = 0.0;
        double sum_confidence = 0.0;
        for (size_t i = 0; i < 8; ++i) {
            sum_weighted += corner_values[i] * corner_confidence[i];
            sum_confidence += corner_confidence[i];
        }
        double local_avg = sum_confidence > 0 ? sum_weighted / sum_confidence : 0.3;
        
        // Blend missing corners toward local average (reduces extreme outliers)
        for (size_t i = 0; i < 8; ++i) {
            if (corner_confidence[i] < 1.0) {  // Missing or low-confidence corner
                // Blend 70% nearest value, 30% local average
                corner_values[i] = 0.7 * corner_values[i] + 0.3 * local_avg;
            }
        }
    }
    
    // Handle degenerate cases (point lies on face, edge, or corner)
    if (x0 == x1 && y0 == y1 && z0 == z1) {
        // Point is exactly on a grid point
        return std::max(0.0, std::min(1.0, corner_values[0]));
    }
    
    if (x0 == x1) {
        // Degenerate in x - do bilinear in yz plane
        if (y0 == y1 && z0 == z1) return std::max(0.0, std::min(1.0, corner_values[0]));
        if (y0 == y1) {
            // Linear in z only
            double frac_z = (double)(z - z0) / (z1 - z0);
            double result = corner_values[0] * (1.0 - frac_z) + corner_values[4] * frac_z;
            return std::max(0.0, std::min(1.0, result));
        }
        if (z0 == z1) {
            // Linear in y only
            double frac_y = (double)(y - y0) / (y1 - y0);
            double result = corner_values[0] * (1.0 - frac_y) + corner_values[2] * frac_y;
            return std::max(0.0, std::min(1.0, result));
        }
        // Bilinear in yz
        double frac_y = (double)(y - y0) / (y1 - y0);
        double frac_z = (double)(z - z0) / (z1 - z0);
        
        double v00 = corner_values[0];
        double v01 = corner_values[4];
        double v10 = corner_values[2];
        double v11 = corner_values[6];
        
        double v0 = v00 * (1.0 - frac_z) + v01 * frac_z;
        double v1 = v10 * (1.0 - frac_z) + v11 * frac_z;
        double result = v0 * (1.0 - frac_y) + v1 * frac_y;
        return std::max(0.0, std::min(1.0, result));
    }
    
    if (y0 == y1) {
        // Degenerate in y - do bilinear in xz plane
        if (z0 == z1) {
            // Linear in x only
            double frac_x = (double)(x - x0) / (x1 - x0);
            double result = corner_values[0] * (1.0 - frac_x) + corner_values[1] * frac_x;
            return std::max(0.0, std::min(1.0, result));
        }
        // Bilinear in xz
        double frac_x = (double)(x - x0) / (x1 - x0);
        double frac_z = (double)(z - z0) / (z1 - z0);
        
        double v00 = corner_values[0];
        double v01 = corner_values[4];
        double v10 = corner_values[1];
        double v11 = corner_values[5];
        
        double v0 = v00 * (1.0 - frac_z) + v01 * frac_z;
        double v1 = v10 * (1.0 - frac_z) + v11 * frac_z;
        double result = v0 * (1.0 - frac_x) + v1 * frac_x;
        return std::max(0.0, std::min(1.0, result));
    }
    
    if (z0 == z1) {
        // Degenerate in z - do bilinear in xy plane (standard bilinear)
        double frac_x = (double)(x - x0) / (x1 - x0);
        double frac_y = (double)(y - y0) / (y1 - y0);
        
        double v00 = corner_values[0];
        double v10 = corner_values[1];
        double v01 = corner_values[2];
        double v11 = corner_values[3];
        
        double v0 = v00 * (1.0 - frac_x) + v10 * frac_x;
        double v1 = v01 * (1.0 - frac_x) + v11 * frac_x;
        double result = v0 * (1.0 - frac_y) + v1 * frac_y;
        return std::max(0.0, std::min(1.0, result));
    }
    
    // Full trilinear interpolation
    double frac_x = (double)(x - x0) / (x1 - x0);
    
    // Y-axis wraps: handle wrap-around interpolation
    int dy = y1 - y0;
    if (dy < 0) dy += dimensionSize;  // Wrapped case (e.g., y0=99, y1=3)
    double frac_y = (dy > 0) ? (double)((y - y0 + dimensionSize) % dimensionSize) / dy : 0.0;
    
    double frac_z = (double)(z - z0) / (z1 - z0);
    
    // Interpolate along x on bottom face (z=z0)
    double c00 = corner_values[0] * (1.0 - frac_x) + corner_values[1] * frac_x;
    double c01 = corner_values[2] * (1.0 - frac_x) + corner_values[3] * frac_x;
    
    // Interpolate along x on top face (z=z1)
    double c10 = corner_values[4] * (1.0 - frac_x) + corner_values[5] * frac_x;
    double c11 = corner_values[6] * (1.0 - frac_x) + corner_values[7] * frac_x;
    
    // Interpolate along y on bottom face
    double c0 = c00 * (1.0 - frac_y) + c01 * frac_y;
    
    // Interpolate along y on top face
    double c1 = c10 * (1.0 - frac_y) + c11 * frac_y;
    
    // Final interpolation along z
    double result = c0 * (1.0 - frac_z) + c1 * frac_z;
    
    // Safety clamp: probabilities must be in [0, 1]
    // This prevents catastrophic squared errors from numerical issues
    return std::max(0.0, std::min(1.0, result));
}

void BilinearModel::findSurroundingPoints3D(int x, int y, int z,
                                             int& x0, int& x1,
                                             int& y0, int& y1,
                                             int& z0, int& z1) const {
    // Use precomputed sorted coordinate lists for O(log m) binary search
    // This is MUCH faster than O(n) iteration through all sampled points
    
    if (!coordsCached || known_x.empty()) {
        // Fallback if coords not cached (shouldn't happen after rebuildKDTree)
        x0 = x1 = y0 = y1 = z0 = z1 = -1;
        return;
    }
    
    // Binary search for surrounding coordinates - O(log m) each
    // where m = number of unique coordinates per dimension (~34 for step=3)
    
    // Find x0, x1 (X-axis does NOT wrap)
    auto it_x = std::lower_bound(known_x.begin(), known_x.end(), x);
    if (it_x != known_x.end() && *it_x == x) {
        x0 = x1 = x;
    } else {
        x1 = (it_x != known_x.end()) ? *it_x : -1;
        x0 = (it_x != known_x.begin()) ? *(--it_x) : -1;
    }
    
    // Find y0, y1 (Y-AXIS WRAPS: y=0 connects to y=dimensionSize-1)
    auto it_y = std::lower_bound(known_y.begin(), known_y.end(), y);
    if (it_y != known_y.end() && *it_y == y) {
        y0 = y1 = y;
    } else {
        y1 = (it_y != known_y.end()) ? *it_y : -1;
        y0 = (it_y != known_y.begin()) ? *(--it_y) : -1;
    }
    
    // Handle Y wrapping: if we're at the edge, wrap around
    if (y0 == -1 && !known_y.empty()) {
        // Below the minimum: wrap to top
        y0 = known_y.back();
    }
    if (y1 == -1 && !known_y.empty()) {
        // Above the maximum: wrap to bottom
        y1 = known_y.front();
    }
    
    // Find z0, z1 (Z-axis does NOT wrap)
    auto it_z = std::lower_bound(known_z.begin(), known_z.end(), z);
    if (it_z != known_z.end() && *it_z == z) {
        z0 = z1 = z;
    } else {
        z1 = (it_z != known_z.end()) ? *it_z : -1;
        z0 = (it_z != known_z.begin()) ? *(--it_z) : -1;
    }
    
    // Handle edges for NON-wrapping axes: clamp to boundary
    if (x0 == -1) x0 = x1;
    if (x1 == -1) x1 = x0;
    if (z0 == -1) z0 = z1;
    if (z1 == -1) z1 = z0;
    
    // For Y-axis: if still -1, fallback
    if (y0 == -1) y0 = y1;
    if (y1 == -1) y1 = y0;
}

int BilinearModel::getAdaptiveK(const std::vector<int>& query) const {
    if (!knnTree || treeData.empty()) {
        return 27;  // Default k
    }
    
    // For smooth continuous functions with step=3 sampling:
    // - We have good spatial coverage (3.9%)
    // - But only 7 samples per point means higher variance
    // - Solution: use MORE neighbors to smooth out variance
    //
    // With step=3, neighbors are ~3-5 units apart
    // A 5×5×5 cube (k=125) covers radius ~7.5 units
    // This provides excellent smoothing for continuous functions
    //
    // For smooth functions, more neighbors = better (averages out noise)
    // For discontinuous functions, fewer neighbors = better (preserves edges)
    //
    // Since visualizations show smooth functions with large regions:
    
    int k = (dimensions == 2) ? 25 : 64;  // 5×5 for 2D, 4×4×4 for 3D
    
    // Ensure k doesn't exceed available data
    return std::min(k, (int)treeData.size());
}

#endif // BILINEAR || TESTING
