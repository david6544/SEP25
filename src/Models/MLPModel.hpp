// MLPModel.cpp
#if defined(MLP) || defined(TESTING)

#include "MLPModel.hpp"

#include <random>
#include <limits>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream> // for optional debug (comment out prints for judged runs)

// single RNG for this translation unit (seeded once)
static std::mt19937 GLOBAL_RNG((unsigned)123456789);

// Helper to create TreeNode with safe defaults
static TreeNode* make_node() {
    TreeNode* node = new TreeNode();
    node->split_dimension = -1;
    node->split_value = 0;
    node->left = node->right = nullptr;
    return node;
}

// -------------------- Constructor --------------------
MLPModel::MLPModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries),
      net(MLPNetwork({dimensions, 64, 64, 1}, {tanh_act, tanh_act, identity})),
      root(nullptr),
      observedMin(std::numeric_limits<double>::infinity()),
      observedMax(-std::numeric_limits<double>::infinity()),
      min_leaf_size(3),
      variance_threshold(0.03)
{
    // nothing else needed here
}

// -------------------- Exploration helpers --------------------
void MLPModel::get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves) {
    if (!node) return;

    if (node->is_leaf()) {
        double totalCoords = 1.0;
        for (int i = 0; i < dimensions; i++) {
            totalCoords *= static_cast<double>(node->max_bound[i] - node->min_bound[i] + 1);
        }

        if ((int)node->points.size() < static_cast<int>(totalCoords)) { // still space to explore
            leaves.push_back(node);
        }
        return;
    }

    get_explore_leaves(node->left, leaves);
    get_explore_leaves(node->right, leaves);
}

std::vector<int> MLPModel::get_next_query() {
    if (!root) return std::vector<int>(dimensions, std::max(0, dimensionSize / 2));

    std::vector<TreeNode*> leaves;
    get_explore_leaves(root, leaves);

    double alpha = 1.0; // exploration weight
    double beta  = 1.0; // exploitation weight

    double best_score = -1e18;
    std::vector<int> best_candidate;

    for (auto leaf : leaves) {
        for (int i = 0; i < 5; ++i) {
            auto candidate = get_random_candidate(leaf);
            // ensure candidate length and values valid
            bool bad = false;
            if ((int)candidate.size() != dimensions) bad = true;
            for (int d = 0; d < dimensions && !bad; ++d) {
                if (candidate[d] < 0 || candidate[d] >= dimensionSize) bad = true;
            }
            if (bad) continue;

            double score = beta * get_exploitation_score(leaf) + alpha * get_exploration_score(leaf);
            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }
    }

    // If nothing found (shouldn't happen), fallback to random unseen
    if (best_candidate.empty()) {
        std::vector<int> fallback(dimensions, 0);
        for (int d = 0; d < dimensions; ++d) fallback[d] = std::min(0, dimensionSize - 1);
        return fallback;
    }

    return best_candidate;
}

std::vector<int> MLPModel::get_random_candidate(TreeNode* leaf) {
    std::vector<int> candidate(dimensions, 0);
    for (int d = 0; d < dimensions; ++d) {
        std::vector<int> dVals;
        dVals.reserve((leaf->points.size() + 2));
        // safe sentinel values clamped to domain
        int minS = std::max(0, leaf->min_bound[d] - 1);
        int maxS = std::min(dimensionSize - 1, leaf->max_bound[d] + 1);
        dVals.push_back(minS);
        for (auto& p : leaf->points) {
            int v = p.coords[d];
            // clamp points as defensive measure
            if (v < 0) v = 0;
            if (v >= dimensionSize) v = dimensionSize - 1;
            dVals.push_back(v);
        }
        dVals.push_back(maxS);
        std::sort(dVals.begin(), dVals.end());
        int best_run_len = -1;
        int best_lo = dVals.front();
        int best_hi = dVals.back();
        for (int i = 1; i < (int)dVals.size(); ++i) {
            int diff = dVals[i] - dVals[i-1];
            if (diff > best_run_len) {
                best_run_len = diff;
                best_lo = dVals[i-1];
                best_hi = dVals[i];
            }
        }
        int mid = (best_lo + best_hi) / 2;
        // clamp to valid range
        if (mid < 0) mid = 0;
        if (mid >= dimensionSize) mid = dimensionSize - 1;
        candidate[d] = mid;
    }
    return candidate;
}

// -------------------- Scoring --------------------
double MLPModel::get_exploitation_score(TreeNode* leaf) {
    if (leaf->points.empty()) return 0.0;

    // Density factor: prefer sparsely explored leaves
    double totalCoords = 1.0;
    for (int i = 0; i < dimensions; i++)
        totalCoords *= static_cast<double>(leaf->max_bound[i] - leaf->min_bound[i] + 1);
    double density_score = 1.0 - (static_cast<double>(leaf->points.size()) / totalCoords);

    // Predicted value at leaf center (use center but ensure valid)
    std::vector<int> center(dimensions);
    for (int i = 0; i < dimensions; i++) {
        center[i] = (leaf->min_bound[i] + leaf->max_bound[i]) / 2;
        center[i] = std::max(0, std::min(center[i], dimensionSize - 1));
    }

    double predicted_value = predict_coordinate(center);

    // If we haven't observed any outputs yet, predicted_value may be a default -> scale accordingly
    if (!std::isfinite(predicted_value)) predicted_value = 0.0;

    return density_score * predicted_value;
}

double MLPModel::get_exploration_score(TreeNode* leaf) {
    if (leaf->points.empty()) return 1.0; // completely unexplored

    // Use model's variance on leaf points (denormalized)
    std::vector<double> preds;
    preds.reserve(leaf->points.size());
    for (auto& p : leaf->points) {
        auto pr = predict_coordinate(p.coords);
        if (std::isfinite(pr)) preds.push_back(pr);
        else preds.push_back(0.0);
    }

    double mean_pred = 0.0;
    for (double v : preds) mean_pred += v;
    mean_pred /= preds.size();

    double var = 0.0;
    for (double v : preds) {
        double diff = v - mean_pred;
        var += diff * diff;
    }
    var /= preds.size();

    // Normalize to [0,1] robustly
    if (!std::isfinite(var)) var = 0.0;
    return var / (var + 1.0);
}

// -------------------- Normalization --------------------
std::vector<double> MLPModel::normalize_input(const std::vector<int>& coords, int dimensionSize_) {
    std::vector<double> x(coords.size(), 0.0);
    double denom = (dimensionSize_ <= 1) ? 1.0 : (dimensionSize_ - 1.0);
    for (size_t i = 0; i < coords.size(); ++i) {
        double v = static_cast<double>(coords[i]) / denom;
        if (!std::isfinite(v)) v = 0.0;
        // clamp to [0,1]
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        x[i] = v;
    }
    return x;
}

double MLPModel::normalize_output(double value) {
    if (!std::isfinite(value)) return 0.5;
    if (observedMax <= observedMin) return 0.5; // avoid division by zero; no data or single-point range
    double v = (value - observedMin) / (observedMax - observedMin);
    if (!std::isfinite(v)) v = 0.5;
    // clamp
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return v;
}

double MLPModel::denormalize_output(double value) {
    if (!std::isfinite(value)) value = 0.5;
    if (observedMax <= observedMin) return observedMin; // unknown range => return observedMin (or 0)
    double v = value * (observedMax - observedMin) + observedMin;
    if (!std::isfinite(v)) v = observedMin;
    return v;
}

// -------------------- Update / Training --------------------
void MLPModel::update_prediction(const std::vector<int>& query, double result) {
    currentQuery++;

    // Update observed range safely
    if (std::isfinite(result)) {
        if (result < observedMin) observedMin = result;
        if (result > observedMax) observedMax = result;
    }

    // If root not built yet, build initial root node
    if (roo
