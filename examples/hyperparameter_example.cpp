/**
 * Example: Using RegionModel with Custom Hyperparameters
 * 
 * This example shows how to create and use different RegionModel configurations
 */

#include "src/Models/RegionModel.hpp"
#include "src/Models/RegionModelPresets.hpp"
#include <iostream>

int main() {
    int dimensions = 3;
    int dimensionSize = 100;
    int totalQueries = 300000;
    
    // Example 1: Using the optimized preset for 100³
    std::cout << "=== Optimized for 100³ Grid ===" << std::endl;
    auto config1 = RegionModelPresets::Optimized100Cube();
    RegionModel model1(dimensions, dimensionSize, totalQueries, config1);
    
    std::cout << "Initial grid step: " << config1.initial_grid_step << std::endl;
    std::cout << "KNN k: " << config1.knn_k << std::endl;
    std::cout << "Discontinuity threshold: " << config1.variance_threshold_discontinuity << std::endl;
    
    // Example 2: Custom configuration for specific needs
    std::cout << "\n=== Custom Configuration ===" << std::endl;
    RegionModelConfig config2;
    
    // Tune for discontinuous functions
    config2.initial_grid_step = 8;
    config2.samples_boundary_walking = 12;
    config2.variance_threshold_discontinuity = 0.12;
    config2.gradient_threshold = 0.25;
    config2.knn_k = 10;
    
    RegionModel model2(dimensions, dimensionSize, totalQueries, config2);
    
    std::cout << "Custom step size: " << config2.initial_grid_step << std::endl;
    std::cout << "Boundary samples: " << config2.samples_boundary_walking << std::endl;
    std::cout << "Gradient threshold: " << config2.gradient_threshold << std::endl;
    
    // Example 3: Budget-conscious approach
    std::cout << "\n=== Budget Saver ===" << std::endl;
    auto config3 = RegionModelPresets::BudgetSaver();
    RegionModel model3(dimensions, dimensionSize, totalQueries, config3);
    
    std::cout << "Sparse grid step: " << config3.initial_grid_step << std::endl;
    std::cout << "Minimal samples: " << config3.samples_region_low_var << std::endl;
    
    // Example 4: Function-specific presets
    std::cout << "\n=== Smooth Functions Preset ===" << std::endl;
    auto smooth_config = RegionModelPresets::SmoothFunctions();
    std::cout << "KNN k for smooth interpolation: " << smooth_config.knn_k << std::endl;
    std::cout << "Reduced samples: " << smooth_config.samples_initial_grid << std::endl;
    
    std::cout << "\n=== Discontinuous Functions Preset ===" << std::endl;
    auto disc_config = RegionModelPresets::DiscontinuousFunctions();
    std::cout << "Boundary samples: " << disc_config.samples_boundary_walking << std::endl;
    std::cout << "Sensitive threshold: " << disc_config.variance_threshold_discontinuity << std::endl;
    
    // Example 5: A/B testing different configurations
    std::cout << "\n=== Configuration Comparison ===" << std::endl;
    
    struct ConfigTest {
        std::string name;
        RegionModelConfig config;
        double estimated_initial_queries;
    };
    
    std::vector<ConfigTest> tests = {
        {"Default", RegionModelPresets::Default(), 0},
        {"Optimized100", RegionModelPresets::Optimized100Cube(), 0},
        {"BudgetSaver", RegionModelPresets::BudgetSaver(), 0},
        {"HighAccuracy", RegionModelPresets::HighAccuracy(), 0}
    };
    
    for (auto& test : tests) {
        int grid_points = dimensionSize / test.config.initial_grid_step;
        int total_grid_points = grid_points * grid_points * grid_points;
        test.estimated_initial_queries = total_grid_points * test.config.samples_initial_grid;
        
        std::cout << test.name << ":" << std::endl;
        std::cout << "  Grid points: " << total_grid_points << std::endl;
        std::cout << "  Initial queries: ~" << test.estimated_initial_queries << std::endl;
        std::cout << "  Budget used: " << (test.estimated_initial_queries / totalQueries * 100) << "%" << std::endl;
    }
    
    return 0;
}
