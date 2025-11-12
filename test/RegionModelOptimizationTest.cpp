#include <gtest/gtest.h>
#include "../src/Models/RegionModel.hpp"
#include <cmath>
#include <vector>

// Test fixture for RegionModel
class RegionModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up common test data
    }
};

// Test 1: Verify faster discontinuity detection with variance
TEST_F(RegionModelTest, DiscontinuityDetectionWithVariance) {
    int dims = 3;
    int size = 20;
    int queries = 2000;
    
    RegionModel model(dims, size, queries);
    
    // Simulate a boundary: half plane with different values
    // Points with x < 10: mostly 0.0, x >= 10: mostly 1.0
    for (int i = 0; i < 50; i++) {
        std::vector<int> query_low = {5, 10, 10};
        std::vector<int> query_high = {15, 10, 10};
        
        // Low side: mostly 0
        double result_low = (i % 10 < 2) ? 1.0 : 0.0;  // 20% ones
        model.update_prediction(query_low, result_low);
        
        // High side: mostly 1
        double result_high = (i % 10 > 7) ? 0.0 : 1.0;  // 80% ones
        model.update_prediction(query_high, result_high);
    }
    
    // After several samples, should detect discontinuity
    // Get predictions (this triggers internal logic)
    double pred_low = model.get_value_at({5, 10, 10});
    double pred_high = model.get_value_at({15, 10, 10});
    
    // Predictions should be different
    EXPECT_LT(pred_low, pred_high);
    EXPECT_LT(pred_low, 0.5);
    EXPECT_GT(pred_high, 0.5);
}

// Test 2: Verify Y-axis wrapping distance
TEST_F(RegionModelTest, YAxisWrappingDistance) {
    int dims = 3;
    int size = 100;
    int queries = 10000;
    
    RegionModel model(dims, size, queries);
    
    // Test wrapped distance calculation
    // Point at y=5 should be closer to y=95 (wrapped) than straight distance
    std::vector<int> point_a = {50, 5, 50};
    std::vector<int> point_b = {50, 95, 50};
    
    // The model should handle this internally in distance_with_wrapping
    // We can't call it directly, but we can verify behavior through predictions
    
    // Sample both points with similar values
    for (int i = 0; i < 10; i++) {
        model.update_prediction(point_a, 0.8);
        model.update_prediction(point_b, 0.8);
    }
    
    // A point at y=0 should get similar prediction due to wrapping
    std::vector<int> point_wrapped = {50, 0, 50};
    double pred = model.get_value_at(point_wrapped);
    
    // Should be close to 0.8 due to nearby sampled points with wrapping
    EXPECT_GT(pred, 0.5);  // Should be influenced by nearby high values
}

// Test 3: Verify adaptive sampling
TEST_F(RegionModelTest, AdaptiveSamplingBehavior) {
    int dims = 3;
    int size = 30;
    int queries = 5000;
    
    RegionModel model(dims, size, queries);
    
    // Query the model and track how many times same point is queried
    std::map<std::vector<int>, int> query_counts;
    std::vector<int> last_query;
    int repeat_count = 0;
    
    // Generate queries and track repeats
    for (int i = 0; i < 100; i++) {
        auto query = model.get_next_query();
        
        if (query == last_query) {
            repeat_count++;
        } else {
            if (repeat_count > 0) {
                query_counts[last_query] = repeat_count + 1;
            }
            last_query = query;
            repeat_count = 0;
        }
        
        // Simulate result
        model.update_prediction(query, 0.5);
    }
    
    // We should see adaptive sampling (different points get different sample counts)
    // Initial grid phase should use ~5 samples
    EXPECT_GT(query_counts.size(), 0);
    
    // At least some points should have been sampled multiple times
    bool has_multi_sample = false;
    for (const auto& kv : query_counts) {
        if (kv.second >= 3) {
            has_multi_sample = true;
            break;
        }
    }
    EXPECT_TRUE(has_multi_sample);
}

// Test 4: Face neighbors optimization
TEST_F(RegionModelTest, FaceNeighborsOnly) {
    int dims = 3;
    int size = 20;
    int queries = 1000;
    
    RegionModel model(dims, size, queries);
    
    // Create a clear discontinuity along x-axis
    // x < 10: value 0.0, x >= 10: value 1.0
    std::vector<int> center = {10, 10, 10};
    
    // Sample center and its 6 face neighbors
    std::vector<std::vector<int>> face_neighbors = {
        {9, 10, 10}, {11, 10, 10},  // X-axis
        {10, 9, 10}, {10, 11, 10},  // Y-axis
        {10, 10, 9}, {10, 10, 11}   // Z-axis
    };
    
    // Sample center with mixed values
    for (int i = 0; i < 10; i++) {
        model.update_prediction(center, (i % 2 == 0) ? 0.0 : 1.0);
    }
    
    // Sample left neighbor (x=9) with low values
    for (int i = 0; i < 10; i++) {
        model.update_prediction(face_neighbors[0], 0.0);
    }
    
    // Sample right neighbor (x=11) with high values  
    for (int i = 0; i < 10; i++) {
        model.update_prediction(face_neighbors[1], 1.0);
    }
    
    // The model should detect this as a boundary
    // Verify predictions reflect the discontinuity
    double pred_left = model.get_value_at({9, 10, 10});
    double pred_right = model.get_value_at({11, 10, 10});
    
    EXPECT_LT(pred_left, 0.3);
    EXPECT_GT(pred_right, 0.7);
}

// Test 5: Phase transitions
TEST_F(RegionModelTest, PhaseTransitions) {
    int dims = 3;
    int size = 20;
    int queries = 5000;  // Enough to trigger phase transitions
    
    RegionModel model(dims, size, queries);
    
    // Run through queries and collect results
    std::vector<double> results;
    for (int i = 0; i < 1000; i++) {
        auto query = model.get_next_query();
        double result = 0.5;  // Constant for simplicity
        model.update_prediction(query, result);
        results.push_back(result);
    }
    
    // Model should survive phase transitions without crashing
    // Just verify we can query successfully
    double pred = model.get_value_at({10, 10, 10});
    EXPECT_GE(pred, 0.0);
    EXPECT_LE(pred, 1.0);
}

// Test 6: Prediction clamping to [0, 1]
TEST_F(RegionModelTest, PredictionClamping) {
    int dims = 3;
    int size = 20;
    int queries = 1000;
    
    RegionModel model(dims, size, queries);
    
    // Sample some points
    for (int i = 0; i < 50; i++) {
        auto query = model.get_next_query();
        model.update_prediction(query, (i % 2 == 0) ? 0.0 : 1.0);
    }
    
    // Query many random points
    for (int x = 0; x < size; x += 5) {
        for (int y = 0; y < size; y += 5) {
            for (int z = 0; z < size; z += 5) {
                double pred = model.get_value_at({x, y, z});
                
                // All predictions should be in valid range
                EXPECT_GE(pred, 0.0) << "Prediction below 0 at (" << x << "," << y << "," << z << ")";
                EXPECT_LE(pred, 1.0) << "Prediction above 1 at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}
