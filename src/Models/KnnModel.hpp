#if defined(KNN) || defined(TESTING)
#ifndef KNN_MODEL_H
#define KNN_MODEL_H

#include "Model.hpp"

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
    double split_value;
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

/**
 * @brief The DumbModel is merely for testing the client - It 
 * randomly picks query points and sets statespace[query point] = returned value
 * 
 */
class KnnModel : public Model {
private:
    TreeNode* root = nullptr;
    int min_leaf_size = 3;
    int variance_threshold = 0.03;
    
    TreeNode* find_leaf(TreeNode* node, const std::vector<int>& query);

    void insert_point(TreeNode* node, const Point& p);

    TreeNode* build_tree(std::vector<Point>& data, const std::vector<int>& min_bound,const std::vector<int>& max_bound);

    double get_variance(std::vector<Point>& data);

    void get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves);

    std::vector<int> get_random_candidate(TreeNode* node);
    
    double get_exploitation_score(TreeNode* leaf);

    double get_exploration_score(TreeNode* leaf);

    double predict_coordinate(const std::vector<int>& coordinate);

    void updateStateSpace(std::vector<int>& coords, int idx);

public:
    /**
     * @brief Construct a new Knn Model object
     * 
     * @param dimensions The number of dimensions
     * @param dimensionSize The size of the dimensions
     * @param totalQueries The total number of queries available
     */
    KnnModel(int dimensions, int dimensionSize, int totalQueries);

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
};


#endif // KNN_MODEL_H
#endif