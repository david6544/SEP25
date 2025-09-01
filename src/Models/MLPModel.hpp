#if defined(MLP) || defined(TESTING)
#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <limits>
#include <cmath>

#include "Model.hpp"
#include "Tools/MLP.hpp"

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

struct RunningStats {
    double mean = 0.0;
    double m2   = 0.0;   // sum of squares of differences from the current mean
    size_t n    = 0;

    void push(double x) {
        n++;
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance() const { return (n > 1) ? m2 / (n - 1) : 1.0; }
    double stddev()   const { return std::sqrt(variance() + 1e-8); }
};

struct Normalizer {
    RunningStats stats;

    double normalize(double y) {
        if (stats.n < 2) return y; // not enough data yet, pass through
        return (y - stats.mean) / stats.stddev();
    }

    double denormalize(double y) {
        if (stats.n < 2) return y; // not enough data yet, pass through
        return y * stats.stddev() + stats.mean;
    }
};

/**
 * @brief The DumbModel is merely for testing the client - It 
 * randomly picks query points and sets statespace[query point] = returned value
 * 
 */
class MLPModel : public Model {
private:
    MLPNetwork net;
    std::vector<Point> seenPoints;
    TreeNode* root = nullptr;
    Normalizer yNorm;
    int min_leaf_size = 3;
    int variance_threshold = 0.03;
    
    TreeNode* find_leaf(TreeNode* node, const std::vector<int>& query);

    void insert_point(TreeNode* node, const std::vector<int>& query, double result);

    TreeNode* build_tree(std::vector<Point>& data, const std::vector<int>& min_bound,const std::vector<int>& max_bound);

    double get_variance(std::vector<Point>& data);

    void get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves);

    std::vector<int> get_random_candidate(TreeNode* node);
    
    double get_exploitation_score(TreeNode* leaf);

    double get_exploration_score(TreeNode* leaf);

    double predict_coordinate(const std::vector<int>& coordinate);

    std::vector<double> normalize_input(const std::vector<int>& coords, int dimensionSize);
    double normalize_output(double value);
    double denormalize_output(double value);

public:
    /**
     * @brief Construct a new MLP Model object
     * 
     * @param dimensions The number of dimensions
     * @param dimensionSize The size of the dimensions
     * @param totalQueries The total number of queries available
     */
    MLPModel(int dimensions, int dimensionSize, int totalQueries);

    /**
     * @brief Get the next query object
     * 
     * @return std::vector<int> the query coordinates
     */
    std::vector<int> get_next_query() override;

    /**
     * @brief updates our model with the queried point and result from the query
     * 
     * @param query coordinates of the query
     * @param result result returned from black box for our query
     */
    void update_prediction(const std::vector<int> &query, double result) override;

    double get_value_at(const std::vector<int> &query) override;
};


#endif // MLP_MODEL_H
#endif