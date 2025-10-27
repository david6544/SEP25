#if defined(REGION) || defined(TESTING)
#include "RegionModel.hpp"
#include <cmath>
#include <algorithm>
#include <set>

RegionModel::RegionModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries) {
    
    // Start with initial grid sampling
    generateInitialGrid();
}

RegionModel::~RegionModel() {
    if (knnTree) {
        delete knnTree;
    }
}

int RegionModel::coordsToIndex(const std::vector<int>& coords) const {
    int index = 0;
    int multiplier = 1;
    for (int i = dimensions - 1; i >= 0; --i) {
        index += coords[i] * multiplier;
        multiplier *= dimensionSize;
    }
    return index;
}

std::vector<int> RegionModel::indexToCoords(int index) const {
    std::vector<int> coords(dimensions);
    for (int i = dimensions - 1; i >= 0; --i) {
        coords[i] = index % dimensionSize;
        index /= dimensionSize;
    }
    return coords;
}

// OPTIMIZATION 3: Adaptive samples per point based on context
int RegionModel::getAdaptiveSamples(const std::vector<int>& point) {
    int idx = coordsToIndex(point);
    auto it = sampledPoints.find(idx);
    
    // Initial grid: 5-7 samples (enough to detect discontinuity)
    if (current_phase == Phase::INITIAL_GRID) {
        return 5;
    }
    
    // Boundary walking: 8-10 samples (high confidence needed)
    if (current_phase == Phase::BOUNDARY_WALKING) {
        return 8;
    }
    
    // Region filling: adaptive based on variance
    if (current_phase == Phase::REGION_FILLING) {
        if (it != sampledPoints.end() && it->second.count > 0) {
            double var = it->second.variance();
            if (var > 0.2) return 10;  // High variance = uncertain
            if (var > 0.1) return 7;   // Medium variance
            return 5;                   // Low variance = smooth region
        }
        return 5;  // Default for new points
    }
    
    // Pattern matching: fewer samples (12-15 for uncertain areas)
    if (current_phase == Phase::PATTERN_MATCHING) {
        if (it != sampledPoints.end() && it->second.count > 0) {
            double var = it->second.variance();
            if (var > 0.2) return 12;  // Very uncertain
            return 3;                   // Already have data
        }
        return 5;
    }
    
    return 5;  // Default fallback
}

// OPTIMIZATION 2: Distance calculation with Y-axis wrapping
double RegionModel::distance_with_wrapping(const std::vector<int>& a, const std::vector<int>& b) const {
    double d = 0.0;
    for (int i = 0; i < dimensions; i++) {
        double diff = std::abs(a[i] - b[i]);
        if (i == 1 && dimensions == 3) {  // Y-axis wraps in 3D
            diff = std::min(diff, dimensionSize - diff);
        }
        d += diff * diff;
    }
    return std::sqrt(d);
}

void RegionModel::generateInitialGrid() {
    // Phase 1: Coarse grid sampling to detect discontinuities
    // Use ~10-20% of budget for initial exploration
    // Step size ~10-15 for good coverage with limited samples
    
    int step = 10;  // 10×10×10 = 1000 points × 3 samples = 3000 queries (~1% of budget)
    
    std::vector<int> point(dimensions);
    std::function<void(int)> generateRecursive = [&](int dim) {
        if (dim == dimensions) {
            queryQueue.push(point);
            return;
        }
        for (int i = 0; i < dimensionSize; i += step) {
            point[dim] = i;
            generateRecursive(dim + 1);
        }
    };
    generateRecursive(0);
    
    // Add boundaries (y-axis wraps, but still sample edges)
    if (dimensions == 3) {
        for (int x : {0, dimensionSize-1}) {
            for (int y = 0; y < dimensionSize; y += step) {
                for (int z : {0, dimensionSize-1}) {
                    queryQueue.push({x, y, z});
                }
            }
        }
    }
}

std::vector<int> RegionModel::get_next_query() {
    currentQuery++;
    
    // Check if we need to move to next point
    if (repeatQueries == 0) {
        // Phase transitions based on query budget usage
        double budget_used = (double)currentQuery / totalQueries;
        
        if (current_phase == Phase::INITIAL_GRID && budget_used > 0.15) {
            // Transition to boundary walking after using 15% of budget
            current_phase = Phase::BOUNDARY_WALKING;
            
            // Find discontinuities from initial grid
            for (const auto& kv : sampledPoints) {
                if (detectDiscontinuity(kv.second.coords)) {
                    boundaryPoints.insert(coordsToIndex(kv.second.coords));
                    queryQueue.push(kv.second.coords);
                }
            }
        }
        else if (current_phase == Phase::BOUNDARY_WALKING && budget_used > 0.40) {
            // Transition to region filling after using 40% of budget
            current_phase = Phase::REGION_FILLING;
            
            // Identify regions and queue interior sampling
            for (const auto& kv : sampledPoints) {
                if (!kv.second.is_boundary) {
                    int region_id = assignRegion(kv.second.coords);
                    // Queue interior points for dense sampling
                }
            }
        }
        else if (current_phase == Phase::REGION_FILLING && budget_used > 0.70) {
            // Transition to pattern matching
            current_phase = Phase::PATTERN_MATCHING;
            findPatterns();
        }
        
        // Get next query point
        if (!queryQueue.empty()) {
            currentPoint = queryQueue.front();
            queryQueue.pop();
            // OPTIMIZATION 3: Use adaptive sampling
            repeatQueries = getAdaptiveSamples(currentPoint);
        } else {
            // Fallback: sample center if queue empty
            currentPoint = std::vector<int>(dimensions, dimensionSize / 2);
            repeatQueries = getAdaptiveSamples(currentPoint);
        }
    }
    
    repeatQueries--;
    return currentPoint;
}

void RegionModel::update_prediction(const std::vector<int>& query, double result) {
    int idx = coordsToIndex(query);
    
    auto& sp = sampledPoints[idx];
    if (sp.coords.empty()) {
        sp.coords = query;
    }
    sp.sum += result;
    sp.sum_squares += result * result;
    sp.count++;
    
    // Mark as visited
    visitedPoints.insert(idx);
    
    // If in boundary walking phase, check neighbors for more boundaries
    if (current_phase == Phase::BOUNDARY_WALKING) {
        // OPTIMIZATION 3: Check after minimum samples (not full samplesPerPoint)
        if (sp.count >= 3 && detectDiscontinuity(query)) {
            sp.is_boundary = true;
            
            // Add unvisited neighbors to queue
            auto neighbors = getNeighbors(query, 2);
            for (const auto& neighbor : neighbors) {
                int nidx = coordsToIndex(neighbor);
                if (visitedPoints.find(nidx) == visitedPoints.end()) {
                    queryQueue.push(neighbor);
                    visitedPoints.insert(nidx);
                }
            }
        }
    }
}

bool RegionModel::detectDiscontinuity(const std::vector<int>& point) {
    // OPTIMIZATION 1: Faster discontinuity detection using variance + face neighbors only
    int idx = coordsToIndex(point);
    auto it = sampledPoints.find(idx);
    if (it == sampledPoints.end() || it->second.count < 3) {
        return false;  // Need minimum samples
    }
    
    // High variance indicates stochastic boundary or discontinuity
    double center_var = it->second.variance();
    if (center_var > 0.15) {
        return true;  // High variance = likely boundary
    }
    
    double center_value = it->second.average();
    
    // Only check 6 face neighbors (not all 26 for 3D) for efficiency
    std::vector<std::vector<int>> faces;
    if (dimensions == 3) {
        int x = point[0], y = point[1], z = point[2];
        faces = {
            {x-1, y, z}, {x+1, y, z},  // X-axis faces
            {x, (y-1+dimensionSize)%dimensionSize, z},  // Y- (wrapped)
            {x, (y+1)%dimensionSize, z},                // Y+ (wrapped)
            {x, y, z-1}, {x, y, z+1}   // Z-axis faces
        };
    } else {
        // For other dimensions, use basic neighbors
        auto neighbors = getNeighbors(point, 1);
        faces = neighbors;
    }
    
    // Check gradient with face neighbors
    for (const auto& neighbor : faces) {
        // Bounds check (Y wraps, so always valid)
        bool valid = true;
        for (int i = 0; i < dimensions; i++) {
            if (i == 1) continue;  // Y wraps
            if (neighbor[i] < 0 || neighbor[i] >= dimensionSize) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        
        int nidx = coordsToIndex(neighbor);
        auto nit = sampledPoints.find(nidx);
        if (nit != sampledPoints.end() && nit->second.count >= 3) {
            double neighbor_value = nit->second.average();
            
            // Discontinuity if difference > threshold
            if (std::abs(center_value - neighbor_value) > 0.3) {
                return true;
            }
        }
    }
    
    return false;
}

std::vector<std::vector<int>> RegionModel::getNeighbors(const std::vector<int>& point, int radius) {
    std::vector<std::vector<int>> neighbors;
    
    if (dimensions == 3) {
        int x = point[0], y = point[1], z = point[2];
        
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dz = -radius; dz <= radius; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    
                    int nx = x + dx;
                    int ny = (y + dy + dimensionSize) % dimensionSize;  // Y wraps
                    int nz = z + dz;
                    
                    // Check bounds (except y which wraps)
                    if (nx >= 0 && nx < dimensionSize && 
                        nz >= 0 && nz < dimensionSize) {
                        neighbors.push_back({nx, ny, nz});
                    }
                }
            }
        }
    }
    
    return neighbors;
}

int RegionModel::assignRegion(const std::vector<int>& point) {
    // Flood-fill style region assignment
    // Points with similar values belong to same region
    int idx = coordsToIndex(point);
    auto it = sampledPoints.find(idx);
    if (it == sampledPoints.end()) return -1;
    
    if (it->second.region_id != -1) {
        return it->second.region_id;
    }
    
    // Find or create region
    double point_value = it->second.average();
    
    // Check if point belongs to existing region (similar value)
    for (auto& region : regions) {
        if (std::abs(point_value - region.avg_value) < 0.2) {
            it->second.region_id = region.id;
            region.interior_points.push_back(point);
            return region.id;
        }
    }
    
    // Create new region
    Region new_region;
    new_region.id = regions.size();
    new_region.avg_value = point_value;
    new_region.interior_points.push_back(point);
    regions.push_back(new_region);
    
    it->second.region_id = new_region.id;
    return new_region.id;
}

void RegionModel::findPatterns() {
    // Compute pattern signatures for each region
    for (auto& region : regions) {
        region.pattern_hash = computePatternHash(region);
    }
    
    // TODO: Pattern matching and replication logic
    // For now, just use region-based prediction
}

double RegionModel::computePatternHash(const Region& region) {
    // Simple hash: combination of avg value, size, and shape
    double hash = region.avg_value * 1000.0;
    hash += region.interior_points.size();
    hash += region.boundary_points.size() * 10.0;
    return hash;
}

double RegionModel::get_value_at(const std::vector<int>& query) {
    int idx = coordsToIndex(query);
    
    // Check if we have exact data
    auto it = sampledPoints.find(idx);
    if (it != sampledPoints.end() && it->second.count > 0) {
        return it->second.average();
    }
    
    // Build KD-tree if needed
    if (!knnTree && !sampledPoints.empty()) {
        rebuildKDTree();
    }
    
    // Strategy 1: Check if query is in a known region
    double prediction = predictFromRegion(query);
    if (prediction >= 0.0) {
        return std::max(0.0, std::min(1.0, prediction));
    }
    
    // Strategy 2: Use nearest neighbor
    prediction = predictFromNearest(query);
    return std::max(0.0, std::min(1.0, prediction));
}

double RegionModel::predictFromRegion(const std::vector<int>& query) {
    // Find nearest sampled point and use its region
    if (!knnTree || treeData.empty()) return -1.0;
    
    auto neighbors = knnTree->getKNearest(query, 5);
    if (neighbors.empty()) return -1.0;
    
    // Find most common region among nearest neighbors
    std::unordered_map<int, int> region_votes;
    for (const auto& neighbor : neighbors) {
        int nidx = coordsToIndex(neighbor.point);
        auto it = sampledPoints.find(nidx);
        if (it != sampledPoints.end() && it->second.region_id >= 0) {
            region_votes[it->second.region_id]++;
        }
    }
    
    if (region_votes.empty()) return -1.0;
    
    // Get region with most votes
    int best_region = -1;
    int max_votes = 0;
    for (const auto& vote : region_votes) {
        if (vote.second > max_votes) {
            max_votes = vote.second;
            best_region = vote.first;
        }
    }
    
    if (best_region >= 0 && best_region < (int)regions.size()) {
        return regions[best_region].avg_value;
    }
    
    return -1.0;
}

double RegionModel::predictFromNearest(const std::vector<int>& query) {
    if (!knnTree || treeData.empty()) {
        return 0.35;  // Global fallback
    }
    
    // Average K nearest neighbors with distance weighting
    int k = 8;
    auto neighbors = knnTree->getKNearest(query, k);
    
    if (neighbors.empty()) return 0.35;
    
    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    
    for (const auto& neighbor : neighbors) {
        // OPTIMIZATION 2: Use wrapped distance for Y-axis
        double dist = distance_with_wrapping(query, neighbor.point);
        
        double weight = 1.0 / (1.0 + dist);
        sum_weighted += neighbor.value * weight;
        sum_weights += weight;
    }
    
    return sum_weights > 0 ? sum_weighted / sum_weights : 0.35;
}

void RegionModel::rebuildKDTree() {
    if (knnTree) {
        delete knnTree;
        knnTree = nullptr;
    }
    
    treeData.clear();
    treeData.reserve(sampledPoints.size());
    
    for (const auto& kv : sampledPoints) {
        if (kv.second.count > 0) {
            treeData.push_back({kv.second.average(), kv.second.coords});
        }
    }
    
    if (!treeData.empty()) {
        knnTree = new KNNTree(treeData);
    }
}

#endif // REGION || TESTING
