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

TEST(TestKDTreeStateSpace, TestBucketInsertionAndNoSplitUnderThreshold) {
    // For 2D, dimensionSize=4 => totalPoints=16 => log2(16)=4
    // So bucketSizeThreshold = 4
    KDTreeStateSpace tree(2, 4);

    // Insert first node → becomes root
    tree.insert({1, 1}, 1.0);
    auto* root = tree.get_root();
    ASSERT_NE(root, nullptr);

    // Force it into bucket mode for testing
    root->isBucket = true;

    // Add up to 4 unique points (threshold)
    tree.insert({2, 2}, 2.0);
    tree.insert({3, 3}, 3.0);
    tree.insert({0, 0}, 4.0);

    // Should still be bucket, not split
    EXPECT_TRUE(root->isBucket);
    EXPECT_EQ(root->bucket.size(), 4);
    EXPECT_TRUE(root->left == nullptr && root->right == nullptr);
}

TEST(TestKDTreeStateSpace, TestBucketTriggersSplitAtThresholdPlusOne) {
    // 2D, dimensionSize=4 => totalPoints=16 => threshold=4
    KDTreeStateSpace tree(2, 4);

    // Insert one node, then make it bucket mode
    tree.insert({1, 1}, 1.0);
    auto* root = tree.get_root();
    root->isBucket = true;

    // Insert 4 (exceeds to threshold) — not bucket
    tree.insert({2, 2}, 2.0);
    tree.insert({3, 3}, 3.0);
    tree.insert({0, 0}, 4.0);
    tree.insert({1, 2}, 5.0);

    auto* newRoot = tree.get_root();
    ASSERT_NE(newRoot, nullptr);
    EXPECT_FALSE(newRoot->isBucket);
    EXPECT_EQ(newRoot->bucket.size(), 0);

    // Insert 5th → exceeds threshold (log2(16)=4) → should trigger split
    tree.insert({2, 3}, 6.0);

    EXPECT_FALSE(newRoot->isBucket); // now internal node
    EXPECT_TRUE(newRoot->left != nullptr || newRoot->right != nullptr);
    EXPECT_TRUE(newRoot->bucket.empty());
}

TEST(TestKDTreeStateSpace, TestBucketSplitAxisSelection) {
    // 2D, dimensionSize=8 => totalPoints=64 => threshold=6
    KDTreeStateSpace tree(2, 8);

    // Insert root and set as bucket
    tree.insert({1, 5}, 1.0);
    auto* root = tree.get_root();
    root->isBucket = true;

    // Insert points that vary only along x-axis
    tree.insert({9, 5}, 2.0);
    tree.insert({5, 5}, 3.0);
    tree.insert({8, 5}, 4.0);
    tree.insert({6, 5}, 5.0);
    tree.insert({7, 5}, 6.0);

    // Exceed threshold (6)
    tree.insert({10, 5}, 7.0);

    // Should have split along axis 0 (x-axis)
    EXPECT_FALSE(root->isBucket);
    EXPECT_EQ(root->axis, 0);
}

TEST(TestKDTreeStateSpace, TestBucketValueUpdateWorks) {
    // 2D, dimensionSize=4 => threshold=4
    KDTreeStateSpace tree(2, 4);

    tree.insert({1, 1}, 1.0);
    auto* root = tree.get_root();
    root->isBucket = true;

    // Add two points
    tree.insert({2, 2}, 2.0);
    tree.insert({3, 3}, 3.0);

    // Update existing point
    tree.insert({2, 2}, 9.9);

    // Value should have updated, not duplicated
    int count22 = 0;
    double val22 = 0.0;
    for (auto* n : root->bucket) {
        if (n->coords == std::vector<int>({2, 2})) {
            val22 = n->value;
            ++count22;
        }
    }
    EXPECT_EQ(count22, 1);
    EXPECT_DOUBLE_EQ(val22, 9.9);
}

TEST(TestKDTreeStateSpace, TestBucketSplitPreservesAllValues) {
    // 2D, dimensions=2 => dimensionSize=4
    KDTreeStateSpace tree(2, 4);
    // 2D, dimensionSize=4 => totalPoints=16 => threshold=4

    // Force root into bucket mode
    tree.insert({0, 0}, 1.0);
    auto* root = tree.get_root();

    EXPECT_TRUE(root->isBucket);

    // Insert multiple points to exceed threshold and force split
    tree.insert({1, 1}, 2.0);
    tree.insert({2, 2}, 3.0);
    tree.insert({3, 3}, 4.0);
    tree.insert({0, 3}, 5.0); // triggers split

    auto* newRoot = tree.get_root();
    EXPECT_FALSE(newRoot->isBucket);

    // Check that all values are retrievable
    std::cout << tree.get({0, 0}) << "\n";
    std::cout << tree.get({1, 1}) << "\n";
    std::cout << tree.get({2, 2}) << "\n";
    std::cout << tree.get({3, 3}) << "\n";
    std::cout << tree.get({0, 3}) << "\n";
    EXPECT_DOUBLE_EQ(tree.get({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(tree.get({1, 1}), 2.0);
    EXPECT_DOUBLE_EQ(tree.get({2, 2}), 3.0);
    EXPECT_DOUBLE_EQ(tree.get({3, 3}), 4.0);
    EXPECT_DOUBLE_EQ(tree.get({0, 3}), 5.0);
}
