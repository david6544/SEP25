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

        // For locality calculations
        int visitCount = 0;    // How many times queries have been made near this node
        double totalError = 0.0; // Cumulative prediction error at this location
        //
        std::vector<TreeNode*> bucket; // optional for bucket nodes
        bool isBucket;

        TreeNode(const std::vector<int>& c, double v, int a)
            : coords(c), value(v), axis(a), left(nullptr), right(nullptr), isBucket(false) {}
    };


    TreeNode* root;
    int maxDepth = -1; // No limit by default

    // Helpers
    TreeNode* insert_recursive(TreeNode* node, const std::vector<int>& coords, double value, int depth);
    double get_recursive(TreeNode* node, const std::vector<int>& coords, int depth) const;
    void destroy(TreeNode* node);
    void KDTreeStateSpace::nearest_recursive(TreeNode* node,
                                         const std::vector<int>& target,
                                         int depth,
                                         TreeNode*& bestNode,
                                         double& bestDist) const;
    

public:
    KDTreeStateSpace(int dimensions, int dimensionSize);

    ~KDTreeStateSpace();

    int get_dimensions() const override;
    int get_dimension_size() const override;
    
    double get(const std::vector<int>& coords) const override;
    double insert(const std::vector<int>& coords, double value);

    TreeNode* get_root();
    std::vector<KDTreeStateSpace::TreeNode*> nearest_neighbor(const std::vector<int>& coords) const;

    double squared_distance(const std::vector<int>& a, const std::vector<int>& b) const;
};

#endif // KD_TREE_STATE_SPACE
