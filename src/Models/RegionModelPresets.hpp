#ifndef REGION_MODEL_PRESETS_HPP
#define REGION_MODEL_PRESETS_HPP

#include "RegionModel.hpp"

/**
 * Preset configurations for RegionModel for different scenarios
 */
namespace RegionModelPresets {

    /**
     * Default configuration (conservative, balanced)
     */
    inline RegionModelConfig Default() {
        return RegionModelConfig();
    }

    /**
     * Optimized for 100³ grid with 300k queries (30% budget)
     * More aggressive initial sampling, reduced samples in smooth regions
     */
    inline RegionModelConfig Optimized100Cube() {
        RegionModelConfig cfg;
        
        // More aggressive initial coverage
        cfg.initial_grid_step = 8;  // 12.5³ ≈ 1953 points
        
        // Reduce samples in early phases
        cfg.samples_initial_grid = 4;           // Was 5
        cfg.samples_boundary_walking = 8;       // Keep high for boundaries
        cfg.samples_region_low_var = 3;         // Was 5 - save queries in smooth regions
        cfg.samples_region_med_var = 6;         // Was 7
        cfg.samples_region_high_var = 10;       // Keep high for uncertainty
        cfg.samples_pattern_low_var = 2;        // Was 3 - minimal for known patterns
        cfg.samples_pattern_high_var = 12;      // Keep high
        
        // More sensitive boundary detection
        cfg.variance_threshold_high = 0.18;     // Was 0.2
        cfg.variance_threshold_medium = 0.09;   // Was 0.1
        cfg.variance_threshold_discontinuity = 0.12;  // Was 0.15 - more aggressive
        cfg.gradient_threshold = 0.25;          // Was 0.3 - more sensitive
        
        // Better predictions with more neighbors
        cfg.knn_k = 12;  // Was 8 - smoother interpolation
        
        return cfg;
    }

    /**
     * Budget-conscious configuration
     * Minimizes queries while maintaining reasonable accuracy
     */
    inline RegionModelConfig BudgetSaver() {
        RegionModelConfig cfg;
        
        // Sparser initial grid
        cfg.initial_grid_step = 12;  // 8.3³ ≈ 571 points
        
        // Minimal samples everywhere
        cfg.samples_initial_grid = 3;
        cfg.samples_boundary_walking = 6;
        cfg.samples_region_low_var = 2;
        cfg.samples_region_med_var = 4;
        cfg.samples_region_high_var = 7;
        cfg.samples_pattern_low_var = 2;
        cfg.samples_pattern_high_var = 8;
        
        // Less aggressive boundary detection
        cfg.variance_threshold_discontinuity = 0.18;
        cfg.gradient_threshold = 0.35;
        
        // Fewer neighbors
        cfg.knn_k = 6;
        
        return cfg;
    }

    /**
     * High accuracy configuration
     * Uses more queries for better precision (suitable for larger budgets)
     */
    inline RegionModelConfig HighAccuracy() {
        RegionModelConfig cfg;
        
        // Dense initial grid
        cfg.initial_grid_step = 7;  // 14.3³ ≈ 2915 points
        
        // More samples everywhere
        cfg.samples_initial_grid = 6;
        cfg.samples_boundary_walking = 12;
        cfg.samples_region_low_var = 6;
        cfg.samples_region_med_var = 10;
        cfg.samples_region_high_var = 15;
        cfg.samples_pattern_low_var = 4;
        cfg.samples_pattern_high_var = 16;
        
        // Very sensitive boundary detection
        cfg.variance_threshold_discontinuity = 0.10;
        cfg.gradient_threshold = 0.20;
        
        // Many neighbors for smooth predictions
        cfg.knn_k = 15;
        cfg.neighbor_search_radius = 3;
        
        return cfg;
    }

    /**
     * For smooth, continuous functions (bowl-shaped, plates)
     * Lower boundary sensitivity, fewer samples
     */
    inline RegionModelConfig SmoothFunctions() {
        RegionModelConfig cfg;
        
        cfg.initial_grid_step = 10;
        
        // Fewer samples since less stochasticity
        cfg.samples_initial_grid = 3;
        cfg.samples_boundary_walking = 5;
        cfg.samples_region_low_var = 2;
        cfg.samples_region_med_var = 4;
        cfg.samples_region_high_var = 7;
        
        // Less sensitive to boundaries (few discontinuities expected)
        cfg.variance_threshold_discontinuity = 0.20;
        cfg.gradient_threshold = 0.40;
        
        // More neighbors for smoother interpolation
        cfg.knn_k = 15;
        
        return cfg;
    }

    /**
     * For highly discontinuous functions (many local minima, steep ridges)
     * Higher boundary sensitivity, more samples at boundaries
     */
    inline RegionModelConfig DiscontinuousFunctions() {
        RegionModelConfig cfg;
        
        cfg.initial_grid_step = 8;
        
        // More samples for boundary confidence
        cfg.samples_initial_grid = 5;
        cfg.samples_boundary_walking = 12;
        cfg.samples_region_low_var = 4;
        cfg.samples_region_med_var = 8;
        cfg.samples_region_high_var = 12;
        
        // Very sensitive boundary detection
        cfg.variance_threshold_discontinuity = 0.10;
        cfg.gradient_threshold = 0.20;
        
        // Moderate neighbors (avoid over-smoothing)
        cfg.knn_k = 8;
        cfg.neighbor_search_radius = 3;
        
        return cfg;
    }

} // namespace RegionModelPresets

#endif // REGION_MODEL_PRESETS_HPP
