#if defined(BOO) || defined(TESTING)

#include "BBOptimizer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>

#include <iostream>
using namespace std;

// ---------------- Point ----------------
Point::Point(const std::vector<int>& c, double v) : coords(c), value(v) {}

// ---------------- TreeNode ----------------
TreeNode::TreeNode() : split_dimension(-1), split_value(-1), left(nullptr), right(nullptr) {}
bool TreeNode::is_leaf() const { return left == nullptr && right == nullptr; }

// ---------------- BBOptimizer ----------------
BBOptimizer::BBOptimizer(int dims, int dimSize, int totalQueries, int seed)
    : Model(dims, dimSize, totalQueries), root(nullptr),
      min_leaf_size(4), variance_threshold(0.01), rd(seed) {}

// --------- Private methods ---------
TreeNode* BBOptimizer::build_tree(std::vector<Point>& data,
                                  const std::vector<int>& min_bound,
                                  const std::vector<int>& max_bound) {
    TreeNode* node = new TreeNode();
    node->min_bound = min_bound;
    node->max_bound = max_bound;

    if (data.size() <= min_leaf_size || get_variance(data) <= variance_threshold) {
        node->points = data;
        return node;
    }

    int best_dim = -1;
    double best_spread = -1;
    for (int dim = 0; dim < dimensions; ++dim) {
        int min_val = data[0].coords[dim];
        int max_val = data[0].coords[dim];
        for (auto& p : data) {
            min_val = std::min(min_val, p.coords[dim]);
            max_val = std::max(max_val, p.coords[dim]);
        }
        double spread = max_val - min_val;
        if (spread > best_spread) {
            best_spread = spread;
            best_dim = dim;
        }
    }

    if (best_spread == 0) {
        node->points = data;
        return node;
    }

    node->split_dimension = best_dim;
    int min_val = data[0].coords[best_dim];
    int max_val = data[0].coords[best_dim];
    for (auto& p : data) {
        min_val = std::min(min_val, p.coords[best_dim]);
        max_val = std::max(max_val, p.coords[best_dim]);
    }
    node->split_value = (min_val + max_val) / 2;

    std::vector<Point> left_pts, right_pts;
    for (auto& p : data) {
        if (p.coords[best_dim] <= node->split_value) left_pts.push_back(p);
        else right_pts.push_back(p);
    }

    if (left_pts.empty() || right_pts.empty()) {
        node->points = data;
        return node;
    }

    std::vector<int> left_max = max_bound;
    left_max[best_dim] = node->split_value;
    std::vector<int> right_min = min_bound;
    right_min[best_dim] = node->split_value + 1;

    node->left = build_tree(left_pts, min_bound, left_max);
    node->right = build_tree(right_pts, right_min, max_bound);

    return node;
}

double BBOptimizer::get_variance(const std::vector<Point>& data) {
    if (data.empty()) return 1.0;
    double mean = 0.0;
    for (auto& p : data) mean += p.value;
    mean /= data.size();
    double var = 0.0;
    for (auto& p : data) {
        double d = p.value - mean;
        var += d*d;
    }
    var /= data.size();
    return var;
}

TreeNode* BBOptimizer::find_leaf(TreeNode* node, const std::vector<int>& query) {
    if (!node || node->is_leaf()) return node;
    if (query[node->split_dimension] <= node->split_value)
        return find_leaf(node->left, query);
    else
        return find_leaf(node->right, query);
}

double BBOptimizer::leaf_value_variance(TreeNode* leaf) {
    if (!leaf || leaf->points.size() < 2) return 1.0;
    double mean = 0.0;
    for (auto& p : leaf->points) mean += p.value;
    mean /= leaf->points.size();
    double var = 0.0;
    for (auto& p : leaf->points) {
        double d = p.value - mean;
        var += d*d;
    }
    var /= leaf->points.size() - 1;
    return std::max(1e-6, var);
}

double BBOptimizer::leaf_nn_distance(TreeNode* leaf, const std::vector<int>& x) {
    if (!leaf || leaf->points.empty()) {
        double s = 0;
        for (int d=0; d<dimensions; ++d)
            s += std::pow((leaf? (leaf->max_bound[d]-leaf->min_bound[d]) : (dimensionSize-1)),2);
        return std::sqrt(s);
    }
    double best = std::numeric_limits<double>::infinity();
    for (auto& p: leaf->points) {
        double dist = 0;
        for (int i=0;i<dimensions;i++){
            double d = x[i]-p.coords[i];
            dist += d*d;
        }
        dist = std::sqrt(dist);
        if (dist<best) best = dist;
    }
    return best;
}

std::vector<int> BBOptimizer::get_random_candidate(TreeNode* leaf) {
    std::vector<int> candidate(dimensions);
    for (int d=0; d<dimensions; ++d) {
        int minv = leaf->min_bound[d];
        int maxv = leaf->max_bound[d];
        std::uniform_int_distribution<int> dist(minv, maxv);
        candidate[d] = dist(rd);
    }
    return candidate;
}

// --------- Public methods ---------
std::vector<int> BBOptimizer::get_next_query() {
    // --- 1) No tree yet: return center ---
    if (root == nullptr) 
        return std::vector<int>(dimensions, (dimensionSize / 2.0));

    // --- 2) Collect all leaf nodes ---
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack{root};
    while (!stack.empty()) {
        TreeNode* node = stack.back(); stack.pop_back();
        if (!node) continue;
        if (node->is_leaf()) leaves.push_back(node);
        else {
            stack.push_back(node->left);
            stack.push_back(node->right);
        }
    }

    // --- 3) Try random candidates from leaves ---
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_candidate;

    for (auto leaf : leaves) {
        for (int tries = 0; tries < 5; ++tries) {
            auto candidate = get_random_candidate(leaf);
            if (value_cache.find(candidate) != value_cache.end()) continue;

            double mean_val = 0.0;
            for (auto& p : leaf->points) mean_val += p.value;
            if (!leaf->points.empty()) mean_val /= leaf->points.size();

            double sigma = std::sqrt(leaf_value_variance(leaf)) * (1.0 + leaf_nn_distance(leaf, candidate));
            double score = mean_val + sigma;

            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }
    }

    // --- 4) Fallback: pick center of a leaf ---
    if (best_candidate.empty()) {
        for (auto leaf : leaves) {
            std::vector<int> fallback(dimensions);
            for (int d = 0; d < dimensions; ++d) {
                fallback[d] = (leaf->min_bound[d] + leaf->max_bound[d]) / 2;
            }
            if (value_cache.find(fallback) == value_cache.end())
                return fallback;
        }
    }

    // --- 5) Absolute last resort: global center ---
    if (best_candidate.empty())
        best_candidate = std::vector<int>(dimensions, dimensionSize / 2);

    return best_candidate;
}


void BBOptimizer::update_prediction(const std::vector<int>& query, double result) {
    value_cache[query] = result;
    seenPoints.emplace_back(query,result);

    std::vector<int> min_bound(dimensions,0);
    std::vector<int> max_bound(dimensions,dimensionSize-1);
    root = build_tree(seenPoints, min_bound, max_bound);
}

double BBOptimizer::get_value_at(const std::vector<int>& query) {
    if (value_cache.find(query) != value_cache.end())
        return value_cache[query];

    TreeNode* leaf = find_leaf(root, query);
    if (!leaf || leaf->points.empty()) return 0.0;
    double mean = 0.0;
    for (auto& p: leaf->points) mean += p.value;
    return mean / leaf->points.size();
}

#endif 