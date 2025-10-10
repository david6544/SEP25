#include <iostream>
#include <gtest/gtest.h>

#include "../src/StateSpace/ArrayStateSpace.hpp"
#include "../src/StateSpace/KDTreeStateSpace.hpp"

TEST(TestStateSpace, TestsGetAndSetIn1D){
    ArrayStateSpace mySpace(1, 5);
    ASSERT_EQ(0.0, mySpace.get({1}));
    mySpace.set({1}, -1);
    ASSERT_EQ(-1.0, mySpace.get({1}));
}

TEST(TestStateSpace, TestsGetAndSetInND){
    for (int i = 1; i < 6; i++){
        ArrayStateSpace mySpace(i, 10);
        EXPECT_EQ(i, mySpace.get_dimensions());
        EXPECT_EQ(10, mySpace.get_dimension_size());

        std::vector<int> point(i, i);
    
        EXPECT_EQ(0.0, mySpace.get(point));
        EXPECT_NO_THROW(mySpace.set(point, 23));
        EXPECT_EQ(23.0, mySpace.get(point));
    }
}

TEST(TestStateSpace, TestsErrorThrowing){
    ArrayStateSpace mySpace(1, 5);

    EXPECT_THROW(mySpace.get({6}), std::out_of_range);

    EXPECT_THROW(mySpace.get({-1}), std::out_of_range);

    EXPECT_THROW(mySpace.get({5, 5}), std::invalid_argument);


    ArrayStateSpace mySpace2(4, 4);
    
    EXPECT_THROW(mySpace2.get({1,1,1,5}), std::out_of_range);
    EXPECT_THROW(mySpace2.set({1,1,1,5}, 1), std::out_of_range);

    EXPECT_THROW(mySpace2.get({5, 5}), std::invalid_argument);

}

// KDTreeStateSpace tests

// ---------- Basic insert / get tests ----------

TEST(TestKDTreeStateSpace, TestInsertAndGetIn1D) {
    KDTreeStateSpace tree(1, 5);

    // Initially unset value should be 0.0 (default return)
    EXPECT_DOUBLE_EQ(0.0, tree.get({2}));

    // Insert and retrieve
    tree.insert({2}, 3.14);
    EXPECT_DOUBLE_EQ(3.14, tree.get({2}));

    // Update existing node
    tree.insert({2}, -1.5);
    EXPECT_DOUBLE_EQ(-1.5, tree.get({2}));
}


TEST(TestKDTreeStateSpace, TestInsertAndGetInND) {
    for (int dim = 1; dim <= 3; ++dim) {
        KDTreeStateSpace tree(dim, 10);

        EXPECT_EQ(dim, tree.get_dimensions());
        EXPECT_EQ(10, tree.get_dimension_size());

        std::vector<int> coords(dim, 4);

        EXPECT_DOUBLE_EQ(0.0, tree.get(coords));

        tree.insert(coords, 7.25);
        EXPECT_DOUBLE_EQ(7.25, tree.get(coords));

        // Reinsert with new value
        tree.insert(coords, 9.99);
        EXPECT_DOUBLE_EQ(9.99, tree.get(coords));
    }
}


// ---------- Error handling ----------

TEST(TestKDTreeStateSpace, TestDimensionalityMismatch) {
    KDTreeStateSpace tree(2, 5);

    // Wrong dimensionality
    EXPECT_THROW(tree.insert({1, 2, 3}, 1.0), std::invalid_argument);
    EXPECT_THROW(tree.get({1, 2, 3}), std::invalid_argument);
}


TEST(TestKDTreeStateSpace, TestOutOfRangeReturnsDefault) {
    KDTreeStateSpace tree(2, 5);

    // Nothing inserted yet, should return 0.0 rather than throw
    EXPECT_DOUBLE_EQ(0.0, tree.get({4, 4}));

    // Insert something else and ensure unrelated coords still return default
    tree.insert({1, 1}, 42.0);
    EXPECT_DOUBLE_EQ(42.0, tree.get({1, 1}));
    EXPECT_DOUBLE_EQ(0.0, tree.get({0, 0}));
}


// ---------- Nearest neighbor tests ----------

TEST(TestKDTreeStateSpace, TestNearestNeighborExactMatch) {
    KDTreeStateSpace tree(2, 10);
    tree.insert({2, 3}, 1.5);
    tree.insert({7, 8}, 2.5);
    tree.insert({5, 5}, 3.5);

    auto* nn = tree.nearest_neighbor({5, 5});
    ASSERT_NE(nullptr, nn);
    EXPECT_EQ(nn->coords, std::vector<int>({5, 5}));
    EXPECT_DOUBLE_EQ(nn->value, 3.5);
}


TEST(TestKDTreeStateSpace, TestNearestNeighborClosestPoint) {
    KDTreeStateSpace tree(2, 10);
    tree.insert({0, 0}, 1.0);
    tree.insert({2, 2}, 2.0);
    tree.insert({9, 9}, 9.0);

    // Query point closer to (2,2)
    auto* nn = tree.nearest_neighbor({3, 3});
    ASSERT_NE(nullptr, nn);
    EXPECT_EQ(nn->coords, std::vector<int>({2, 2}));
    EXPECT_DOUBLE_EQ(nn->value, 2.0);
}


// ---------- Bucket depth behavior ----------

TEST(TestKDTreeStateSpace, TestDeepInsertionBuckets) {
    KDTreeStateSpace tree(3, 10);

    // Force many inserts to reach maxDepth
    for (int i = 0; i < 100; ++i) {
        tree.insert({i % 10, (i * 3) % 10, (i * 7) % 10}, i * 1.0);
    }

    // Ensure get returns correct value for one inserted element
    EXPECT_DOUBLE_EQ(90.0, tree.get({0, 0, 0})); // default value, might not exist
    tree.insert({1, 1, 1}, 123.456);
    EXPECT_DOUBLE_EQ(123.456, tree.get({1, 1, 1}));
}


// ---------- Stress / stability test ----------

TEST(TestKDTreeStateSpace, TestMultipleInsertsAndGets) {
    KDTreeStateSpace tree(2, 20);

    // Insert grid
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            double val = x * 10 + y;
            tree.insert({x, y}, val);
        }
    }

    // Retrieve values
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            double expected = x * 10 + y;
            EXPECT_DOUBLE_EQ(expected, tree.get({x, y}));
        }
    }
}