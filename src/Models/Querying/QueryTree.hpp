#pragma once
#include <vector>
#include <set>
#include <unordered_set>
#include <random>
#include <limits>
#include "../Tools/CandidateGenerator.hpp"

struct TreeNode {
    TreeNode *left = nullptr, *right = nullptr, *parent = nullptr;
    int splitDim = -1, splitValue = -1;
    std::vector<std::pair<double, std::vector<int>>> points;
    std::vector<std::vector<int>> dimensionLimits;
    CandidateGenerator *cg;

    ~TreeNode(){
        delete cg;
    }
};

class QueryTree {
public:
    QueryTree(int dims, int dimSize, int leafSize);
    ~QueryTree();
    std::vector<int> get_next_query();
    void update_prediction(const std::vector<int>& query, double result);
    std::pair<std::vector<int>, TreeNode*> nextLeaf;
private:
    TreeNode* root;
    std::set<TreeNode*> leaves;
    int dims, dimSize, leafSize;
    std::mt19937 rng{std::random_device{}()};

    TreeNode* find_leaf(TreeNode* node, const std::vector<int>& query);

    std::vector<int> get_candidate(TreeNode* leaf);
    bool point_in_leaf(const std::vector<int>& query, const TreeNode* leaf);
};
