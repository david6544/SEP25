#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <functional>

/**
 * @brief The coordinates of the point (queried) and the value at that queried location
 * 
 */
struct Point {
    std::vector<int> coords; // coordinates in state space
    double value;               // observed value (query result)
};

/**
 * @brief a KD tree node
 * 
 */
struct TreeNode {
    int split_dimension;
    int split_value;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
    
    std::vector<Point> points;

    std::vector<int> min_bound;
    std::vector<int> max_bound;
    
    /**
     * @brief check whether node is a leaf node
     * 
     * @return true is a leaf node
     * @return false is NOT a leaf node
     */
    bool is_leaf() const {
        return left == nullptr && right == nullptr;
    }
};

class KD_Tree {
public:
    static TreeNode* build_tree(std::vector<Point>& data, const std::vector<int>& min_bound,const std::vector<int>& max_bound);

    static TreeNode* find_leaf(TreeNode* node, const std::vector<int>& query);

    static void insert_point(TreeNode* node, const std::vector<int>& query, double result);

    static double get_variance(std::vector<Point>& data);

    static void get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves);
};

#endif