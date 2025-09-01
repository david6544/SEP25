#if defined(MLP) || defined(TESTING)
#include <vector>
#include <algorithm>
#include <cmath>
#include <random> 

#include <iostream>

#include "MLPModel.hpp"

// public methods

MLPModel::MLPModel(int dimensions, int dimensionSize, int queries) 
    : Model(dimensions, dimensionSize, queries), 
    net(MLPNetwork({dimensions, 64, 64, 1}, {tanh_act, tanh_act, identity})) {
    
}

void MLPModel::get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves) {
    if (!node) return;

    if (node->is_leaf()) {
        long long totalCoords = 1;
        for (int i = 0; i < dimensions; i++)
            totalCoords *= (node->max_bound[i] - node->min_bound[i] + 1);

        if ((int)node->points.size() < totalCoords) { // still space to explore
            leaves.push_back(node);
        }
        return;
    }

    get_explore_leaves(node->left, leaves);
    get_explore_leaves(node->right, leaves);
}


std::vector<int> MLPModel::get_next_query() {
    if (!root) return std::vector<int>(dimensions, dimensionSize / 2);

    std::vector<TreeNode*> leaves;
    get_explore_leaves(root, leaves);

    double alpha = 1.0; // exploration weight
    double beta  = 1.0; // exploitation weight

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
        dVals.reserve((leaf->points.size() + 2));
        dVals.push_back(leaf->min_bound[d]-1);
        for (auto& p : leaf->points) 
            dVals.push_back(p.coords[d]);
        dVals.push_back(leaf->max_bound[d]+1);
        sort(dVals.begin(), dVals.end());
        int run[3] = {0,0,0};
        for (int i = 1; i < dVals.size(); i++){
            int diff = dVals[i] - dVals[i-1];
            if (diff > run[0]){
                run[0] = diff;
                run[1] = dVals[i-1];
                run[2] = dVals[i];
            }
        }
        candidate[d] = (run[1] + run[2])/2;
    }
    return candidate;
}


double MLPModel::get_exploitation_score(TreeNode* leaf) {
    if (leaf->points.empty()) return 0.0;

    // Density factor: prefer sparsely explored leaves
    long long totalCoords = 1;
    for (int i = 0; i < dimensions; i++)
        totalCoords *= (leaf->max_bound[i] - leaf->min_bound[i] + 1);
    double density_score = 1.0 - ((double)leaf->points.size() / totalCoords);

    // Predicted value at leaf center
    std::vector<int> center(dimensions);
    for (int i = 0; i < dimensions; i++)
        center[i] = (leaf->min_bound[i] + leaf->max_bound[i]) / 2;

    double predicted_value = predict_coordinate(center);

    return density_score * predicted_value;
}

/**
 * @brief get the variance of the coordinates high variance indicates a level of exploration 
 * 
 * @param leaf the leaf node in question
 * @return double the score of the exploration metric
 */
double MLPModel::get_exploration_score(TreeNode* leaf) {
    if (leaf->points.empty()) return 1.0; // completely unexplored

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

    return var / (var + 1.0); // normalized to [0,1]
}


std::vector<double> MLPModel::normalize_input(const std::vector<int>& coords, int dimensionSize) {
    std::vector<double> x(coords.size());
    for (int i = 0; i < coords.size(); ++i)
        x[i] = coords[i] / (dimensionSize - 1.0);
    return x;
}

double MLPModel::normalize_output(double value) {
    if (observedMax == observedMin) return 0.5; // avoid division by zero
    return (value - observedMin) / (observedMax - observedMin);
}

double MLPModel::denormalize_output(double value) {
    return value * (observedMax - observedMin) + observedMin;
}


void MLPModel::update_prediction(const std::vector<int>& query, double result) {
    currentQuery++;

    // Update observed range
    observedMin = std::min(observedMin, result);
    observedMax = std::max(observedMax, result);

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
    net.train_sample(x, y, 0.02);

    // Train on all seen points for multiple epochs
    std::random_device rd;
    std::mt19937 g(rd());
    for (int epoch = 0; epoch < 3; ++epoch) {
        std::shuffle(seenPoints.begin(), seenPoints.end(), g);
        for (auto &p : seenPoints) {
            std::vector<double> px = normalize_input(p.coords, dimensionSize);
            std::vector<double> py{normalize_output(p.value)};
            net.train_sample(px, py, 0.02);
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
    double y_pred = net.predict(x)[0];
    return denormalize_output(y_pred);
}


double MLPModel::get_value_at(const std::vector<int> &query){
    return predict_coordinate(query);
}
#endif