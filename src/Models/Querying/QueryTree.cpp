#include "QueryTree.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>

QueryTree::QueryTree(int dims, int dimSize, int leafSize)
    : dims(dims), dimSize(dimSize), leafSize(leafSize) 
{
    root = new TreeNode;
    root->dimensionLimits = std::vector<std::vector<int>>(dims, {0, dimSize-1});
    root->cg = new CandidateGenerator(dims, root->dimensionLimits);
    leaves.insert(root);
}

void freeTree(TreeNode* node){
    if (!node) return;
    
    if (node->left) freeTree(node->left);
    if (node->right) freeTree(node->right);

    delete node->cg;
    node->cg = nullptr;

    delete node;
}


QueryTree::~QueryTree(){
    freeTree(root);
}

bool QueryTree::point_in_leaf(const std::vector<int>& query, const TreeNode* leaf) {
    for (size_t d = 0; d < query.size(); ++d)
        if (query[d] < leaf->dimensionLimits[d][0] || query[d] > leaf->dimensionLimits[d][1])
            return false;
    return true;
}

std::vector<int> QueryTree::get_candidate(TreeNode* leaf) {
    if (leaf->points.empty()) {
        std::vector<int> mid(dims);
        for (int d = 0; d < dims; ++d)
            mid[d] = (leaf->dimensionLimits[d][0] + leaf->dimensionLimits[d][1]) / 2;
        return mid;
    }

    std::vector<std::vector<int>> candidates = leaf->cg->getCandidates(5);

    std::vector<int> bestPoint;
    double bestScore = -1.0;

    for (auto& candidate : candidates) {
        double minDist = std::numeric_limits<double>::infinity();
        for (auto& p : leaf->points) {
            double dist = 0.0;
            for (int d = 0; d < dims; ++d)
                dist += std::abs(candidate[d] - p.second[d]);
            if (dist < minDist) minDist = dist;
        }

        double score = minDist / (1.0 + 0.1 * leaf->points.size());

        if (score > bestScore) {
            bestScore = score;
            bestPoint = candidate;
        }
    }

    if (bestPoint.empty()) {
        bestPoint.resize(dims);
        for (int d = 0; d < dims; ++d)
            bestPoint[d] = (leaf->dimensionLimits[d][0] + leaf->dimensionLimits[d][1]) / 2;
    }

    return bestPoint;
}



std::vector<int> QueryTree::get_next_query() {
    std::pair<double, std::vector<int>> bestChoice = {-1.0, {}};

    for (auto leaf : leaves) {
        std::vector<int> candidate = get_candidate(leaf);

        double score;
        if (!leaf->points.empty()) {
            double minDist = std::numeric_limits<double>::infinity();
            for (auto& p : leaf->points) {
                double dist = 0.0;
                for (int d = 0; d < dims; ++d) dist += std::abs(candidate[d] - p.second[d]);
                if (dist < minDist) minDist = dist;
            }
            score = minDist / (1.0 + 0.1 * leaf->points.size());
        } else {
            score = 1e9;
        }

        if (score > bestChoice.first) {
            bestChoice = {score, candidate};
            nextLeaf = {candidate, leaf};
        }
    }

    if (bestChoice.second.empty()) bestChoice.second = std::vector<int>(dims, dimSize / 2);

    return bestChoice.second;
}

TreeNode* QueryTree::find_leaf(TreeNode* node, const std::vector<int>& query) {
    if (!node->left && !node->right)
        return node;

    if (node->splitDim == -1)
        return node;

    int splitDim = node->splitDim;
    int splitValue = node->splitValue;

    if (query[splitDim] <= splitValue && node->left)
        return find_leaf(node->left, query);
    else if (node->right)
        return find_leaf(node->right, query);
    else
        return node; // fallback in case structure is incomplete
}


void QueryTree::update_prediction(const std::vector<int>& query, double result) {
    TreeNode* target = nullptr;
    if (nextLeaf.second != nullptr && query == this->nextLeaf.first)
        target = nextLeaf.second;
    else
        target = find_leaf(root, query);

    if (!target) target = root;

    try {
        target->cg->addQueriedPoint(query);
    } catch (const std::runtime_error& e) {
        std::cerr << "Warning: query outside leaf limits, skipping addQueriedPoint.\n";
    }

    target->points.push_back({result, query});

    if ((int)target->points.size() <= leafSize) return;

    int n = target->points.size();
    int bestDim = -1;
    double bestVar = -1.0;

    for (int d = 0; d < dims; ++d) {
        std::vector<int> vals;
        vals.reserve(n);
        for (auto& p : target->points) vals.push_back(p.second[d]);

        int minVal = *std::min_element(vals.begin(), vals.end());
        int maxVal = *std::max_element(vals.begin(), vals.end());

        if (maxVal <= minVal) continue;  // Cannot split this dimension

        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
        double var = 0.0;
        for (auto v : vals) var += (v - mean) * (v - mean);
        var /= n;

        if (var > bestVar) {
            bestVar = var;
            bestDim = d;
        }
    }

    if (bestDim == -1) return;

    std::sort(target->points.begin(), target->points.end(),
              [bestDim](auto &a, auto &b){ return a.second[bestDim] < b.second[bestDim]; });

    int mid = n / 2;
    int splitValue = target->points[mid].second[bestDim];

    int parentMin = target->dimensionLimits[bestDim][0];
    int parentMax = target->dimensionLimits[bestDim][1];

    splitValue = std::max(parentMin, std::min(splitValue, parentMax - 1));

    if (splitValue < parentMin || splitValue >= parentMax) return;

    auto left = new TreeNode();
    auto right = new TreeNode();
    left->parent = right->parent = target;

    left->dimensionLimits = right->dimensionLimits = target->dimensionLimits;
    left->dimensionLimits[bestDim][1] = splitValue;
    right->dimensionLimits[bestDim][0] = splitValue + 1;

    if (left->dimensionLimits[bestDim][0] <= left->dimensionLimits[bestDim][1])
        left->cg = new CandidateGenerator(dims, left->dimensionLimits);
    if (right->dimensionLimits[bestDim][0] <= right->dimensionLimits[bestDim][1])
        right->cg = new CandidateGenerator(dims, right->dimensionLimits);

    for (auto& p : target->points) {
        if (p.second[bestDim] <= splitValue && left) left->points.push_back(p);
        else if (right) right->points.push_back(p);
    }

    target->splitDim = bestDim;
    target->splitValue = splitValue;
    target->left = left;
    target->right = right;

    leaves.erase(target);
    if (left) leaves.insert(left);
    if (right) leaves.insert(right);

    if (nextLeaf.second == target) nextLeaf = {{}, nullptr};
    target->points.clear();
}


