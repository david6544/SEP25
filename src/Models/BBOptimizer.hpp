#if defined(BOO) || defined(TESTING)

#ifndef BB_OPTIMIZER_HPP
#define BB_OPTIMIZER_HPP

#include <vector>
#include <map>
#include <random>

#include "Model.hpp"

struct Point {
    std::vector<int> coords;
    double value;
    Point(const std::vector<int>& c, double v);
};

struct TreeNode {
    std::vector<int> min_bound, max_bound;
    int split_dimension;
    int split_value;
    std::vector<Point> points;
    TreeNode* left;
    TreeNode* right;

    TreeNode();
    bool is_leaf() const;
};

class BBOptimizer : public Model {
private:
    TreeNode* root;
    int min_leaf_size;
    double variance_threshold;

    std::map<std::vector<int>, double> value_cache;
    std::vector<Point> seenPoints;
    std::mt19937 rd;

    TreeNode* build_tree(std::vector<Point>& data,
                         const std::vector<int>& min_bound,
                         const std::vector<int>& max_bound);

    double get_variance(const std::vector<Point>& data);
    TreeNode* find_leaf(TreeNode* node, const std::vector<int>& query);
    double leaf_value_variance(TreeNode* leaf);
    double leaf_nn_distance(TreeNode* leaf, const std::vector<int>& x);
    std::vector<int> get_random_candidate(TreeNode* leaf);

public:
    BBOptimizer(int dims, int dimSize, int totalQueries, int seed=0);

    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int>& query, double result) override;
    double get_value_at(const std::vector<int>& query) override;
};

#endif // BB_OPTIMIZER_HPP
#endif