#include "QueryTree.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>

QueryTree::QueryTree(int dims, int dimSize, int leafSize)
    : dims(dims), dimSize(dimSize), leafSize(leafSize) 
{
    cg = new CandidateGenerator(dims, dimSize);
    root = new TreeNode;
    root->dimensionLimits = std::vector<std::vector<int>>(dims, {0, dimSize-1});
    leaves.insert(root);
}

void freeTree(TreeNode* node){
    if (node == nullptr) return;
    
    if (node->left) freeTree(node->left);

    if (node->right) freeTree(node->right);

    delete node;
}

QueryTree::~QueryTree(){
    freeTree(root);
    delete cg;
}

bool QueryTree::point_in_leaf(const std::vector<int>& query, const TreeNode* leaf) {
    for (size_t d = 0; d < query.size(); ++d)
        if (query[d] < leaf->dimensionLimits[d][0] || query[d] > leaf->dimensionLimits[d][1])
            return false;
    return true;
}

std::vector<int> QueryTree::get_candidate(TreeNode* leaf) {
    // Empty leaf → geometric center
    if (leaf->points.empty()) {
        std::vector<int> mid(dims);
        for (int d = 0; d < dims; ++d)
            mid[d] = (leaf->dimensionLimits[d][0] + leaf->dimensionLimits[d][1]) / 2;
        return mid;
    }

    // Generate candidates
    std::vector<std::vector<int>> candidates = cg->getCandidates(leaf->dimensionLimits, 5);

    std::vector<int> bestPoint;
    double bestScore = -1.0;

    for (auto& candidate : candidates) {
        // Compute min distance to existing points
        double minDist = std::numeric_limits<double>::infinity();
        for (auto& p : leaf->points) {
            double dist = 0.0;
            for (int d = 0; d < dims; ++d)
                dist += std::abs(candidate[d] - p.second[d]);
            if (dist < minDist) minDist = dist;
        }

        // Adjust score with leaf density
        double score = minDist / (1.0 + 0.1 * leaf->points.size());

        if (score > bestScore) {
            bestScore = score;
            bestPoint = candidate;
        }
    }

    // Fallback
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

        // Score candidate relative to leaf points
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
            // Empty leaf → exploration priority
            score = 1e9;
        }

        if (score > bestChoice.first) {
            bestChoice = {score, candidate};
            nextLeaf = {candidate, leaf};
        }
    }

    // Fallback
    if (bestChoice.second.empty()) bestChoice.second = std::vector<int>(dims, dimSize / 2);

    return bestChoice.second;
}


void QueryTree::update_prediction(const std::vector<int>& query, double result) {
    cg->addCandidate(query);
    // find leaf
    TreeNode* target = nullptr;
    if (query == this->nextLeaf.first)
        target = nextLeaf.second;
    else
        for (auto leaf : leaves)
            if (point_in_leaf(query, leaf)) { target = leaf; break; }
            
    if (!target) target = root;

    target->points.push_back({result, query});

    if ((int)target->points.size() > leafSize) {
        int n = target->points.size();
        int bestDim = 0;
        double bestVar = -1.0;

        for (int d = 0; d < dims; ++d) {
            std::vector<double> vals;
            vals.reserve(n);
            for (auto& p : target->points) vals.push_back(p.second[d]);
            double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
            double var = 0.0;
            for (auto v : vals) var += (v - mean) * (v - mean);
            var /= n;

            if (var > bestVar) { bestVar = var; bestDim = d; }
        }

        std::sort(target->points.begin(), target->points.end(),
                  [bestDim](auto &a, auto &b){ return a.second[bestDim] < b.second[bestDim]; });

        int mid = n/2;
        int splitValue = target->points[mid].second[bestDim];

        auto left = new TreeNode();
        auto right = new TreeNode();
        left->parent = right->parent = target;

        left->dimensionLimits = right->dimensionLimits = target->dimensionLimits;
        left->dimensionLimits[bestDim][1] = splitValue;
        right->dimensionLimits[bestDim][0] = splitValue + 1;

        for (auto& p : target->points) {
            if (p.second[bestDim] <= splitValue) left->points.push_back(p);
            else right->points.push_back(p);
        }

        target->splitDim = bestDim;
        target->splitValue = splitValue;
        target->left = left;
        target->right = right;

        leaves.erase(target);
        leaves.insert(left);
        leaves.insert(right);

        target->points.clear();
    }
}
