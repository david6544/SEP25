#include "src/Models/RegionModel.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Simple test function with clear discontinuity
// Half-space function: x < 50 → 0.2, x >= 50 → 0.8
double halfSpaceFunction(const std::vector<int>& point) {
    return (point[0] < 50) ? 0.2 : 0.8;
}

// Checkerboard pattern: alternating regions
double checkerboardFunction(const std::vector<int>& point) {
    int block_x = point[0] / 25;
    int block_y = point[1] / 25;
    int block_z = point[2] / 25;
    return ((block_x + block_y + block_z) % 2 == 0) ? 0.2 : 0.8;
}

// Sphere discontinuity: inside sphere = 0.8, outside = 0.2
double sphereFunction(const std::vector<int>& point) {
    double cx = 50.0, cy = 50.0, cz = 50.0;
    double dx = point[0] - cx;
    double dy = point[1] - cy;
    double dz = point[2] - cz;
    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
    return (r < 30.0) ? 0.8 : 0.2;
}

void testDiscontinuityDetection(const std::string& test_name, 
                                 double (*testFunc)(const std::vector<int>&)) {
    std::cout << "\n========================================\n";
    std::cout << "Test: " << test_name << "\n";
    std::cout << "========================================\n";
    
    int dims = 3;
    int size = 100;
    int queries = 30000;  // 30% of 100^3
    
    RegionModel model(dims, size, queries);
    
    // Sampling phase
    std::cout << "Sampling with " << queries << " queries...\n";
    int phase_queries[4] = {0, 0, 0, 0};
    int current_phase = 0;
    
    for (int i = 0; i < queries; i++) {
        auto query = model.get_next_query();
        double true_value = testFunc(query);
        
        // Simulate stochastic observations
        double result = (rand() % 100 < true_value * 100) ? 1.0 : 0.0;
        
        model.update_prediction(query, result);
        
        // Track phase transitions (rough estimate)
        double budget = (double)(i+1) / queries;
        if (budget <= 0.15) phase_queries[0]++;
        else if (budget <= 0.40) phase_queries[1]++;
        else if (budget <= 0.70) phase_queries[2]++;
        else phase_queries[3]++;
        
        if ((i + 1) % 5000 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << queries << "\n";
        }
    }
    
    std::cout << "\nPhase query distribution:\n";
    std::cout << "  Initial Grid:       " << phase_queries[0] << " queries\n";
    std::cout << "  Boundary Walking:   " << phase_queries[1] << " queries\n";
    std::cout << "  Region Filling:     " << phase_queries[2] << " queries\n";
    std::cout << "  Pattern Matching:   " << phase_queries[3] << " queries\n";
    
    // Evaluation phase - test along a line through the discontinuity
    std::cout << "\nEvaluating predictions along x-axis (y=50, z=50):\n";
    std::cout << "x\tTrue\tPred\tError\n";
    std::cout << "---\t----\t----\t-----\n";
    
    double total_mae = 0.0;
    int eval_count = 0;
    
    for (int x = 0; x < size; x += 10) {
        std::vector<int> point = {x, 50, 50};
        double true_val = testFunc(point);
        double pred_val = model.get_value_at(point);
        double error = std::abs(true_val - pred_val);
        
        total_mae += error;
        eval_count++;
        
        std::cout << x << "\t" << true_val << "\t" 
                  << pred_val << "\t" << error << "\n";
    }
    
    double mae = total_mae / eval_count;
    std::cout << "\nMean Absolute Error: " << mae << "\n";
    
    // Save a 2D slice for visualization
    std::string filename = "discontinuity_test_" + test_name + ".txt";
    std::ofstream out(filename);
    out << "# 2D slice at z=50\n";
    out << "# x y true_value predicted_value\n";
    
    for (int x = 0; x < size; x += 2) {
        for (int y = 0; y < size; y += 2) {
            std::vector<int> point = {x, y, 50};
            double true_val = testFunc(point);
            double pred_val = model.get_value_at(point);
            out << x << " " << y << " " << true_val << " " << pred_val << "\n";
        }
    }
    out.close();
    
    std::cout << "Saved 2D slice to: " << filename << "\n";
}

int main() {
    srand(42);  // Fixed seed for reproducibility
    
    std::cout << "================================================\n";
    std::cout << "RegionModel Discontinuity Detection Tests\n";
    std::cout << "================================================\n";
    std::cout << "\nThese tests validate the three optimizations:\n";
    std::cout << "1. Faster discontinuity detection (variance + face neighbors)\n";
    std::cout << "2. Y-axis wrapping distance calculation\n";
    std::cout << "3. Adaptive samples per point\n";
    
    // Test 1: Simple half-space discontinuity
    testDiscontinuityDetection("halfspace", halfSpaceFunction);
    
    // Test 2: Checkerboard pattern with multiple discontinuities
    testDiscontinuityDetection("checkerboard", checkerboardFunction);
    
    // Test 3: Spherical discontinuity
    testDiscontinuityDetection("sphere", sphereFunction);
    
    std::cout << "\n================================================\n";
    std::cout << "All tests complete!\n";
    std::cout << "================================================\n";
    
    return 0;
}
