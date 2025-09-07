#if defined(KNN) || defined(TESTING)
#ifndef KNN_MODEL_H
#define KNN_MODEL_H

#include "Model.hpp"
#include "Tools/KDTree.hpp"

/**
 * @brief The DumbModel is merely for testing the client - It 
 * randomly picks query points and sets statespace[query point] = returned value
 * 
 */
class KnnModel : public Model {
private:
    TreeNode* root;
    int min_leaf_size = 3;
    int variance_threshold = 0.03;
    

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

    double get_value_at(const std::vector<int> &query) override;
};


#endif // KNN_MODEL_H
#endif