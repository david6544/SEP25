#include "QueryTree.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

QueryTree::QueryTree(int dims, int dimSize, int leafSize)
    : dims(dims), dimSize(dimSize), leafSize(leafSize), totalQueries(0)
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

// Compute L2 (Euclidean) distance between two points
double QueryTree::compute_l2_distance(const std::vector<int>& a, const std::vector<int>& b) {
    double sum = 0.0;
    for (int d = 0; d < dims; ++d) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Compute leaf volume for normalization
double QueryTree::compute_leaf_volume(const TreeNode* leaf) {
    double volume = 1.0;
    for (int d = 0; d < dims; ++d) {
        volume *= (leaf->dimensionLimits[d][1] - leaf->dimensionLimits[d][0] + 1);
    }
    return volume;
}

// Estimate local uncertainty as variance of results in the leaf
double QueryTree::estimate_local_uncertainty(const TreeNode* leaf) {
    if (leaf->points.size() < 2) {
        // High uncertainty when few points: encourages exploration
        return 1.0;
    }
    
    double mean = 0.0;
    for (const auto& p : leaf->points) {
        mean += p.first;  // p.first is the result value
    }
    mean /= leaf->points.size();
    
    double variance = 0.0;
    for (const auto& p : leaf->points) {
        double diff = p.first - mean;
        variance += diff * diff;
    }
    variance /= leaf->points.size();
    
    return variance;
}

std::vector<int> QueryTree::get_candidate(TreeNode* leaf) {
    if (leaf->points.empty()) {
        // No points yet: return center of leaf
        std::vector<int> mid(dims);
        for (int d = 0; d < dims; ++d)
            mid[d] = (leaf->dimensionLimits[d][0] + leaf->dimensionLimits[d][1]) / 2;
        return mid;
    }

    // Adaptive candidate count: more candidates for denser leaves
    int candidateCount = std::clamp(5 + (int)std::sqrt(leaf->points.size()), 5, 50);
    std::vector<std::vector<int>> candidates = leaf->cg->getCandidates(candidateCount);

    std::vector<int> bestPoint;
    double bestScore = -1.0;
    
    // Compute alpha: decay from 0.7 (exploration) to 0.3 (exploitation)
    // as totalQueries increases
    double alpha = 0.7 - 0.4 * std::min(1.0, totalQueries / 200.0);
    
    // Estimate local uncertainty for this leaf
    double localUncertainty = estimate_local_uncertainty(leaf);
    
    // Compute normalization factor for distances
    double maxPossibleDist = 0.0;
    for (int d = 0; d < dims; ++d) {
        int range = leaf->dimensionLimits[d][1] - leaf->dimensionLimits[d][0];
        maxPossibleDist += range * range;
    }
    maxPossibleDist = std::sqrt(maxPossibleDist);
    if (maxPossibleDist < 1e-9) maxPossibleDist = 1.0;

    for (auto& candidate : candidates) {
        // Find minimum L2 distance to existing points (exploitation term)
        double minDist = std::numeric_limits<double>::infinity();
        for (auto& p : leaf->points) {
            double dist = compute_l2_distance(candidate, p.second);
            if (dist < minDist) minDist = dist;
        }
        
        // Normalize distance to [0, 1]
        double normalizedDist = minDist / maxPossibleDist;
        
        // Hybrid score: balance exploration (uncertainty) and exploitation (distance)
        double score = alpha * localUncertainty + (1.0 - alpha) * normalizedDist;

        if (score > bestScore) {
            bestScore = score;
            bestPoint = candidate;
        }
    }

    if (bestPoint.empty()) {
        // Fallback to center
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
            // Compute hybrid exploration-exploitation score
            double alpha = 0.7 - 0.4 * std::min(1.0, totalQueries / 200.0);
            double localUncertainty = estimate_local_uncertainty(leaf);
            
            // Find minimum L2 distance to existing points
            double minDist = std::numeric_limits<double>::infinity();
            for (auto& p : leaf->points) {
                double dist = compute_l2_distance(candidate, p.second);
                if (dist < minDist) minDist = dist;
            }
            
            // Normalize distance by leaf dimensions
            double maxPossibleDist = 0.0;
            for (int d = 0; d < dims; ++d) {
                int range = leaf->dimensionLimits[d][1] - leaf->dimensionLimits[d][0];
                maxPossibleDist += range * range;
            }
            maxPossibleDist = std::sqrt(maxPossibleDist);
            if (maxPossibleDist < 1e-9) maxPossibleDist = 1.0;
            
            double normalizedDist = minDist / maxPossibleDist;
            
            // Hybrid score
            score = alpha * localUncertainty + (1.0 - alpha) * normalizedDist;
            
            // Normalize by leaf volume to avoid bias toward large leaves
            double leafVolume = compute_leaf_volume(leaf);
            double volumeNormalization = std::pow(leafVolume, 1.0 / dims);
            score /= volumeNormalization;
            
        } else {
            // Empty leaf: prioritize with high score to encourage exploration
            score = 1e9;
        }

        if (score > bestChoice.first) {
            bestChoice = {score, candidate};
            nextLeaf = {candidate, leaf};
        }
    }

    if (bestChoice.second.empty()) {
        bestChoice.second = std::vector<int>(dims, dimSize / 2);
    }

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
        return node;
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
    totalQueries++;  // Track total queries for alpha decay

    if ((int)target->points.size() <= leafSize) return;

    // Split criterion: use variance in RESULT VALUES rather than coordinates
    // This aligns tree structure with function behavior
    int n = target->points.size();
    int bestDim = -1;
    double bestSplitQuality = -1.0;

    for (int d = 0; d < dims; ++d) {
        std::vector<int> vals;
        vals.reserve(n);
        for (auto& p : target->points) vals.push_back(p.second[d]);

        int minVal = *std::min_element(vals.begin(), vals.end());
        int maxVal = *std::max_element(vals.begin(), vals.end());

        if (maxVal <= minVal) continue;  // Cannot split this dimension

        // Sort points by this dimension to evaluate split quality
        auto sortedPoints = target->points;
        std::sort(sortedPoints.begin(), sortedPoints.end(),
                  [d](auto &a, auto &b){ return a.second[d] < b.second[d]; });

        int midIdx = n / 2;
        
        // Compute variance reduction in results if we split at median
        double leftMean = 0.0, rightMean = 0.0;
        for (int i = 0; i < midIdx; ++i) leftMean += sortedPoints[i].first;
        for (int i = midIdx; i < n; ++i) rightMean += sortedPoints[i].first;
        leftMean /= midIdx;
        rightMean /= (n - midIdx);
        
        double leftVar = 0.0, rightVar = 0.0;
        for (int i = 0; i < midIdx; ++i) {
            double diff = sortedPoints[i].first - leftMean;
            leftVar += diff * diff;
        }
        for (int i = midIdx; i < n; ++i) {
            double diff = sortedPoints[i].first - rightMean;
            rightVar += diff * diff;
        }
        
        // Split quality: weighted variance reduction (lower is better)
        double splitQuality = -(leftVar + rightVar) / n;

        if (splitQuality > bestSplitQuality) {
            bestSplitQuality = splitQuality;
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