#ifndef KD_TREE_STATE_SPACE
#define KD_TREE_STATE_SPACE

#include <vector>
#include <iostream>
#include <queue>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "StateSpace.hpp"

class KDTreeStateSpace : public StateSpace {
private:
    struct TreeNode {
        std::vector<int> coords;
        double value;
        int axis;
        TreeNode* left;
        TreeNode* right;

        std::vector<TreeNode*> bucket; // optional for bucket nodes
        bool isBucket;

        TreeNode(const std::vector<int>& c, double v, int a)
            : coords(c), value(v), axis(a), left(nullptr), right(nullptr), isBucket(false) {}
    };


    TreeNode* root;
    int bucketSizeThreshold = 0; // No limit by default
    std::unordered_set<TreeNode*> leafNodes;

    // Helpers
    TreeNode* insert_recursive(TreeNode* node, const std::vector<int>& coords, double value, int depth);
    double get_recursive(TreeNode* node, const std::vector<int>& coords, int depth) const;
    void destroy(TreeNode* node);
    void nearest_recursive(TreeNode* node,
                                         const std::vector<int>& target,
                                         int depth,
                                         TreeNode*& bestNode,
                                         double& bestDist) const;
    std::string coordsToString(std::vector<int> coords) const;
    
public:
    KDTreeStateSpace(int dimensions, int dimensionSize);

    ~KDTreeStateSpace();

    int get_dimensions() const override;
    int get_dimension_size() const override;
    
    double get(const std::vector<int>& coords) const override;
    double insert(const std::vector<int>& coords, double value);

    TreeNode* get_root();
    KDTreeStateSpace::TreeNode* nearest_neighbor(const std::vector<int>& coords) const;

    double squared_distance(const std::vector<int>& a, const std::vector<int>& b) const;

    std::vector<TreeNode*> get_leaves() const {
        std::vector<TreeNode*> leafNodesVec;
        for (const auto& node : leafNodes) {
            leafNodesVec.push_back(node);
        }
        return leafNodesVec;
    }
};

#endif // KD_TREE_STATE_SPACE
