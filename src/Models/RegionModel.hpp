#if defined(REGION) || defined(TESTING)
#ifndef REGION_MODEL_H
#define REGION_MODEL_H

/**
 * @file RegionModel.hpp
 * @brief Region-based model for discontinuity detection and pattern replication.
 *
 * @details
 * High-level overview:
 * The RegionModel is designed to discover and map structure in a discrete
 * state-space by explicitly detecting discontinuities (boundaries) and
 * characterizing interior regions. It proceeds in phases: an initial coarse
 * grid sampling to detect large-scale features, boundary walking to trace
 * discontinuities, region filling to sample interiors with adaptive sampling
 * based on observed variance, and an optional pattern-matching phase to
 * replicate discovered patterns across similar regions.
 *
 * Relation to the state-space mapping challenge:
 * - Prioritizes early detection of discontinuities so that sampling budget is
 *   focused on boundaries where interpolation would fail.
 * - Uses adaptive sampling counts per phase and per-point variance to allocate
 *   more samples where uncertainty is high.
 * - Integrates a KD-tree for spatial queries when interpolating or assigning
 *   region membership; distances account for axis wrapping where applicable.
 *
 * Public contract: generate queries with `get_next_query()`, update samples
 * with `update_prediction(...)`, and obtain estimates via `get_value_at(...)`.
 *
 * @author David
 * @date 2025-11-02
 */

#include "Model.hpp"
#include "Tools/KNNAlgorithm.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * Configuration parameters for RegionModel hyperparameters
 */
struct RegionModelConfig {
    // Initial grid sampling
    int initial_grid_step = 10;              // Grid spacing for initial sampling
    
    // Adaptive samples per phase
    int samples_initial_grid = 5;            // Samples during initial grid phase
    int samples_boundary_walking = 8;        // Samples during boundary detection
    int samples_region_low_var = 5;          // Samples for low variance regions
    int samples_region_med_var = 7;          // Samples for medium variance regions
    int samples_region_high_var = 10;        // Samples for high variance regions
    int samples_pattern_low_var = 3;         // Samples for known patterns
    int samples_pattern_high_var = 12;       // Samples for uncertain patterns
    
    // Variance thresholds
    double variance_threshold_high = 0.2;    // High variance threshold
    double variance_threshold_medium = 0.1;  // Medium variance threshold
    double variance_threshold_discontinuity = 0.15;  // Early discontinuity detection
    
    // Discontinuity detection
    double gradient_threshold = 0.3;         // Value difference for discontinuity
    int min_samples_for_discontinuity = 3;   // Minimum samples before checking
    
    // KNN prediction
    int knn_k = 8;                          // Number of nearest neighbors
    int neighbor_search_radius = 2;         // Radius for boundary neighbor search
    
    // Default constructor with current values
    RegionModelConfig() = default;
};

/**
 * RegionModel - Discontinuity Detection and Pattern Replication
 * 
 * Strategy:
 * 1. Initial sampling to detect discontinuities
 * 2. Boundary walking to map region edges
 * 3. Region filling to characterize interiors
 * 4. Pattern matching and replication across state space
 */
class RegionModel : public Model {
public:
    /**
     * @brief Construct a new RegionModel with default configuration
     *
     * @param dimensions number of dimensions (e.g., 2 or 3)
     * @param dimensionSize size of each dimension
     * @param totalQueries total query budget
     */
    RegionModel(int dimensions, int dimensionSize, int totalQueries);

    /**
     * @brief Construct a new RegionModel with custom configuration
     *
     * @param dimensions number of dimensions
     * @param dimensionSize size of each dimension
     * @param totalQueries total query budget
     * @param config custom RegionModelConfig to override defaults
     */
    RegionModel(int dimensions, int dimensionSize, int totalQueries, const RegionModelConfig& config);

    /**
     * @brief Destroy the RegionModel
     */
    ~RegionModel();

    /**
     * @brief Get the next point to query according to the current phase and policy.
     *
     * The method manages phase transitions (initial grid -> boundary walking ->
     * region filling -> pattern matching) and returns coordinates for the
     * experiment driver to evaluate.
     *
     * @return std::vector<int> coordinate of the next query
     */
    std::vector<int> get_next_query() override;

    /**
     * @brief Record an observed value for a coordinate.
     *
     * Aggregates samples, marks visited points, triggers boundary detection in
     * the boundary walking phase, and updates region bookkeeping.
     *
     * @param query coordinate vector sampled
     * @param result observed probability value in [0,1]
     */
    void update_prediction(const std::vector<int>& query, double result) override;

    /**
     * @brief Return the model's current estimate for a coordinate.
     *
     * The model will prefer region-based predictions when available, then
     * nearest-neighbour based predictions. If no data exists, returns a
     * neutral fallback.
     *
     * @param query coordinate to estimate
     * @return double estimated probability in [0,1]
     */
    double get_value_at(const std::vector<int>& query) override;

private:
    struct SamplePoint {
        std::vector<int> coords;
        double sum = 0.0;
        double sum_squares = 0.0;
        int count = 0;
        bool is_boundary = false;  // Point lies on discontinuity boundary
        int region_id = -1;        // Which region this point belongs to
        
        double average() const {
            return count > 0 ? sum / count : -1.0;
        }
        
        double variance() const {
            if (count < 2) return 0.0;
            double mean = sum / count;
            return (sum_squares / count) - (mean * mean);
        }
    };

    struct Region {
        int id;
        double avg_value = 0.0;
        double value_range = 0.0;  // max - min within region
        std::vector<std::vector<int>> boundary_points;
        std::vector<std::vector<int>> interior_points;
        std::vector<int> centroid;
        
        // Pattern signature for matching
        double pattern_hash = 0.0;
    };

    // Sampling phases
    enum class Phase {
        INITIAL_GRID,        // Coarse grid to find discontinuities
        BOUNDARY_WALKING,    // Trace discontinuity boundaries
        REGION_FILLING,      // Sample region interiors
        PATTERN_MATCHING     // Find and replicate patterns
    };

    Phase current_phase = Phase::INITIAL_GRID;
    
    // Configuration
    RegionModelConfig config;
    
    // Data storage
    std::unordered_map<int, SamplePoint> sampledPoints;
    std::vector<Region> regions;
    KNNTree* knnTree = nullptr;
    std::vector<std::pair<double, std::vector<int>>> treeData;
    
    // Query management
    int currentQuery = 0;
    std::queue<std::vector<int>> queryQueue;  // Points to sample next
    std::vector<int> currentPoint;
    int samplesPerPoint = 3;  // Stochastic samples per location (adaptive)
    int repeatQueries = 0;
    
    // Adaptive sampling parameters
    int getAdaptiveSamples(const std::vector<int>& point);
    
    // Boundary detection
    std::unordered_set<int> boundaryPoints;
    std::unordered_set<int> visitedPoints;
    
    // Helper functions
    int coordsToIndex(const std::vector<int>& coords) const;
    std::vector<int> indexToCoords(int index) const;
    double distance_with_wrapping(const std::vector<int>& a, const std::vector<int>& b) const;
    
    // Phase-specific logic
    void generateInitialGrid();
    bool detectDiscontinuity(const std::vector<int>& point);
    void walkBoundary(const std::vector<int>& start);
    void fillRegion(int region_id);
    void findPatterns();
    
    // Region analysis
    std::vector<std::vector<int>> getNeighbors(const std::vector<int>& point, int radius = 1);
    int assignRegion(const std::vector<int>& point);
    double computePatternHash(const Region& region);
    
    // Prediction
    double predictFromRegion(const std::vector<int>& query);
    double predictFromNearest(const std::vector<int>& query);
    void rebuildKDTree();
};

#endif // REGION_MODEL_H
#endif // REGION || TESTING
