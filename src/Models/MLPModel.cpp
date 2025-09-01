#if defined(MLP) || defined(TESTING)
#include <vector>
#include <algorithm>
#include <cmath>
#include <random> 
#include <climits>

#include "MLPModel.hpp"

// public methods

MLPModel::MLPModel(int dimensions, int dimensionSize, int queries) 
    : Model(dimensions, dimensionSize, queries), 
    net(MLPNetwork({dimensions, 64, 64, 1}, {tanh_act, tanh_act, ident})) {
    
}

// --- 1) safer get_explore_leaves (avoid overflow) ---
void MLPModel::get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves) {
    if (!node) return;

    if (node->is_leaf()) {
        // compute volume in 64-bit and guard against overflow
        unsigned long long totalCoords = 1ULL;
        bool overflow = false;
        for (int i = 0; i < dimensions; ++i) {
            unsigned long long span = (unsigned long long)(node->max_bound[i] - node->min_bound[i] + 1);
            if (span == 0) { overflow = true; break; }
            if (totalCoords > ULLONG_MAX / span) { overflow = true; break; }
            totalCoords *= span;
        }

        // if overflow, treat as "large" (so leaf is considered explorable if not full)
        if (overflow) {
            if ((size_t)node->points.size() < SIZE_MAX) { // effectively always true
                leaves.push_back(node);
            }
        } else {
            if ((unsigned long long)node->points.size() < totalCoords) {
                leaves.push_back(node);
            }
        }
        return;
    }

    get_explore_leaves(node->left, leaves);
    get_explore_leaves(node->right, leaves);
}


#include <iostream>
using namespace std;

std::vector<int> MLPModel::get_next_query() {
    if (!root) return std::vector<int>(dimensions, dimensionSize / 2);

    std::vector<TreeNode*> leaves;
    get_explore_leaves(root, leaves);

    double alpha = 1.5 * (((double) currentQuery)/totalQueries); // exploration weight
    double beta  = (1 - (((double) currentQuery)/totalQueries)); // exploitation weight

    double best_score = -1e9;
    std::vector<int> best_candidate;

    for (auto leaf : leaves) {
        for (int i = 0; i < 5; ++i) {
            auto candidate = get_random_candidate(leaf);
            double score = beta * get_exploitation_score(leaf) + alpha * get_exploration_score(leaf);
            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }
    }
    return best_candidate;
}

std::vector<int> MLPModel::get_random_candidate(TreeNode* leaf) {
    std::vector<int> candidate(dimensions);
    for (int d = 0; d < dimensions; ++d) {
        std::vector<int> dVals;
        dVals.reserve(leaf->points.size() + 2);
        // Use sentinels but keep them within a reasonable range:
        dVals.push_back(leaf->min_bound[d] - 1);
        for (auto& p : leaf->points)
            dVals.push_back(p.coords[d]);
        dVals.push_back(leaf->max_bound[d] + 1);
        std::sort(dVals.begin(), dVals.end());

        int bestDiff = -1;
        int left = dVals.front(), right = dVals.back();
        for (size_t i = 1; i < dVals.size(); ++i) {
            int diff = dVals[i] - dVals[i-1];
            if (diff > bestDiff) {
                bestDiff = diff;
                left = dVals[i-1];
                right = dVals[i];
            }
        }
        // pick midpoint, then clamp to the leaf bounds
        long mid = ((long)left + (long)right) / 2;
        if (mid < leaf->min_bound[d]) mid = leaf->min_bound[d];
        if (mid > leaf->max_bound[d]) mid = leaf->max_bound[d];
        candidate[d] = (int)mid;
    }
    return candidate;
}



double MLPModel::get_exploitation_score(TreeNode* leaf) {
    if (leaf->points.empty()) return 0.0;

    double totalCoords = 1.0;
    for (int i = 0; i < dimensions; i++)
        totalCoords *= (leaf->max_bound[i] - leaf->min_bound[i] + 1);

    double density_score = 1.0 - ((double)leaf->points.size() / totalCoords);

    std::vector<int> center(dimensions);
    for (int i = 0; i < dimensions; i++)
        center[i] = (leaf->min_bound[i] + leaf->max_bound[i]) / 2;

    double predicted_value = predict_coordinate(center);

    return abs(density_score * predicted_value);
}

double MLPModel::get_exploration_score(TreeNode* leaf) {
    if (leaf->points.empty()) return 1.0;

    double mean_pred = 0.0;
    for (auto& p : leaf->points)
        mean_pred += predict_coordinate(p.coords);
    mean_pred /= leaf->points.size();

    double var = 0.0;
    for (auto& p : leaf->points) {
        double diff = predict_coordinate(p.coords) - mean_pred;
        var += diff * diff;
    }
    var /= leaf->points.size();

    return abs(var / (var + 1.0));
}


std::vector<double> MLPModel::normalize_input(const std::vector<int>& coords, int dimensionSize) {
    std::vector<double> x(coords.size());
    for (int i = 0; i < coords.size(); ++i)
        x[i] = coords[i] / (dimensionSize - 1.0);
    return x;
}

double MLPModel::normalize_output(double value) {
    return yNorm.normalize(value);
}

double MLPModel::denormalize_output(double value) {
    return yNorm.denormalize(value);
}


void MLPModel::update_prediction(const std::vector<int>& query, double result) {
    currentQuery++;

    yNorm.stats.push(result);
    if (root == nullptr) {
        std::vector<Point> data = { Point{query, result} };
        std::vector<int> min_bound(dimensions, 0);
        std::vector<int> max_bound(dimensions, dimensionSize - 1);
        root = build_tree(data, min_bound, max_bound);
    } else {
        // Insert point into tree
        insert_point(root, query, result);
    }

    // Store for online training
    seenPoints.push_back(Point(query, result));

    // Normalize input/output
    std::vector<double> x = normalize_input(query, dimensionSize);
    std::vector<double> y{normalize_output(result)};

    // Train on current point
    if (seenPoints.size() < 0.1 * pow(dimensionSize, dimensions)){
        net.train_sample(x, y, 0.01);
    } else {
        net.train_sample(x, y, 0.01);
        // Train on all seen points for multiple epochs
        std::random_device rd;
        std::mt19937 g(rd());
        for (int epoch = 0; epoch < 4 + (((double)currentQuery)/totalQueries) * 46; ++epoch) {
            std::shuffle(seenPoints.begin(), seenPoints.end(), g);
            for (int i = 0; i < 32 && i < seenPoints.size(); i++) {
                std::vector<double> px = normalize_input(seenPoints[i].coords, dimensionSize);
                std::vector<double> py{normalize_output(seenPoints[i].value)};
                net.train_sample(px, py, 0.01);
            }
        }
    }
}


TreeNode* MLPModel::build_tree(std::vector<Point>& data,
                               const std::vector<int>& min_bound,
                               const std::vector<int>& max_bound) {
    TreeNode* node = new TreeNode();
    node->min_bound = min_bound;
    node->max_bound = max_bound;

    // Stop splitting if leaf criteria met
    if (data.size() <= min_leaf_size || get_variance(data) <= variance_threshold) {
        node->points = data;
        return node;
    }

    // Find dimension with largest spread to split on
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

    if (best_spread == 0) {  // all points identical
        node->points = data;
        return node;
    }

    node->split_dimension = best_dim;

    // split at midpoint
    double min_val = data[0].coords[best_dim];
    double max_val = data[0].coords[best_dim];
    for (auto& p : data) {
        if (p.coords[best_dim] < min_val) min_val = p.coords[best_dim];
        if (p.coords[best_dim] > max_val) max_val = p.coords[best_dim];
    }

    node->split_value = (min_val + max_val) / 2;

    // Partition points into left and right
    std::vector<Point> left_pts, right_pts;
    for (auto& p : data) {
        if (p.coords[best_dim] <= node->split_value) left_pts.push_back(p);
        else right_pts.push_back(p);
    }

    // Left child bounds
    std::vector<int> left_max = max_bound;
    left_max[best_dim] = node->split_value;

    // Right child bounds
    std::vector<int> right_min = min_bound;
    right_min[best_dim] = node->split_value + 1;

    // After partitioning points into left_pts and right_pts:
    if (left_pts.empty() || right_pts.empty()) {
        node->points = data;        // keep as a leaf; this split is not useful
        node->left = node->right = nullptr;
        return node;
    }

    // Else proceed to build children
    node->left  = build_tree(left_pts,  min_bound, left_max);
    node->right = build_tree(right_pts, right_min, max_bound);


    return node;
}


// Find leaf for a query
TreeNode* MLPModel::find_leaf(TreeNode* node, const std::vector<int>& query) {
    if (node->is_leaf()) return node;
    if (query[node->split_dimension] <= node->split_value)
        return find_leaf(node->left, query);
    else
        return find_leaf(node->right, query);
}

double MLPModel::get_variance(std::vector<Point>& data){
    if (data.empty()) return 1.0; // completely unexplored

    double avgPred = 0.0;
    for (auto& p : data) avgPred += p.value;
    avgPred /= data.size();

    double predVar = 0.0;
    for (auto& p : data) {
        double diff = p.value - avgPred;
        predVar += diff * diff;
    }
    predVar /= data.size();

    // Normalize by some max variance
    return predVar / (1.0 + predVar);
}

// Insert point and split if necessary
void MLPModel::insert_point(TreeNode* node, const std::vector<int>& query, double result) {
    if (!node) {
        return;
    }

    if (node->is_leaf()) {
        node->points.push_back(Point(query, result));
        if ((int)node->points.size() > min_leaf_size) {
            std::vector<Point> pts = node->points;
            node->points.clear();

            TreeNode* new_node = build_tree(pts, node->min_bound, node->max_bound);
            if (new_node) {
                node->split_dimension = new_node->split_dimension;
                node->split_value = new_node->split_value;
                node->left = new_node->left;
                node->right = new_node->right;
                new_node->left = new_node->right = nullptr;
                delete new_node;
            }
        }
    } else {
        if (query[node->split_dimension] <= node->split_value){
            insert_point(node->left, query, result);
        }else{
            insert_point(node->right, query, result);
        }
    }
}

double MLPModel::predict_coordinate(const std::vector<int>& coordinate) {
    std::vector<double> x = normalize_input(coordinate, dimensionSize);
    return denormalize_output(net.predict(x)[0]);
}


double MLPModel::get_value_at(const std::vector<int> &query){
    return predict_coordinate(query);
}
#endif