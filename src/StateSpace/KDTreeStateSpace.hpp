#ifndef KD_TREE_STATE_SPACE
#define KD_TREE_STATE_SPACE

#include <vector>
#include <iostream>

#include "StateSpace.hpp"

class KDTreeStateSpace : public StateSpace {
private:
    struct TreeNode {
        std::vector<int> coords;
        double value;
        int axis;
        TreeNode* left;
        TreeNode* right;

        TreeNode(const std::vector<int>& c, double v, int a)
            : coords(c), value(v), axis(a), left(nullptr), right(nullptr) {}
    };

    TreeNode* root;

    // Helpers
    TreeNode* insert_recursive(TreeNode* node, const std::vector<int>& coords, double value, int depth);
    double get_recursive(TreeNode* node, const std::vector<int>& coords, int depth) const;
    void destroy(TreeNode* node);
    void nearest_recursive(TreeNode* node, const std::vector<int>& target, int depth, std::vector<int>& best_coords, double& best_value, double& best_dist) const;
    double squared_distance(const std::vector<int>& a, const std::vector<int>& b) const;

public:
    KDTreeStateSpace(int dimensions, int dimensionSize);

    ~KDTreeStateSpace();

    int get_dimensions() const override;
    int get_dimension_size() const override;
    
    double get(const std::vector<int>& coords) const override;
    double insert(const std::vector<int>& coords, double value);

    TreeNode* get_root();
    std::pair<std::vector<int>, double> nearest_neighbor(const std::vector<int>& coords) const;
};

#endif // KD_TREE_STATE_SPACE
