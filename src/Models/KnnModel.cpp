#if defined(KNN) || defined(TESTING)
#include <vector>
#include <algorithm>
#include <iostream>

#include "KnnModel.hpp"

// public methods

KnnModel::KnnModel(int dimensions, int dimensionSize, int queries) : Model(dimensions, dimensionSize, queries) {
    
}

void KnnModel::get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves) {
    if (node->is_leaf()){
        if (get_exploration_score(node) > 0.1)
            leaves.push_back(node);
        return;
    }
    get_explore_leaves(node->left, leaves);
    get_explore_leaves(node->right, leaves);
}

std::vector<int> KnnModel::get_next_query() {
    if (root == nullptr) {
        // first query: midpoint
        return std::vector<int>(dimensions, dimensionSize / 2);
    }


    std::vector<TreeNode*> leaves;
    get_explore_leaves(root, leaves);

    int candidates_per_leaf = 5;
    
    double best_score = -1e9;
    std::vector<int> best_candidate;

    for (auto leaf : leaves) {
        for (int i = 0; i < candidates_per_leaf; ++i) {
            auto candidate = get_random_candidate(leaf);
            double score = get_exploitation_score(leaf) + 1.0 * get_exploration_score(leaf); // lambda=1
            if (best_score < score){
                best_score = score;
                best_candidate = candidate;
            }
        }
    }
    
    return best_candidate;
}

std::vector<int> KnnModel::get_random_candidate(TreeNode* leaf) {
    // trust that we will get a valid candidate
    while (true){
        // generate candidate
        std::vector<int> candidate(dimensions);
        for (int i = 0; i < dimensions; i++)
            candidate[i] = leaf->min_bound[i] + rand() % (leaf->max_bound[i] - leaf->min_bound[i] + 1);
        
        // check that candidate is not already present
        if (leaf->points.end() == std::find_if(leaf->points.begin(), leaf->points.end(), [candidate](const Point& p) { return (p.coords == candidate);}))
            return candidate;
    }
}


double KnnModel::get_exploitation_score(TreeNode* leaf) {
    long long totalCoordinates = 1;
    for (int i = 0; i < dimensions; i++){
        totalCoordinates *= (leaf->max_bound[i]-leaf->min_bound[i] + 1);
    }
    return ((double) totalCoordinates -  leaf->points.size()) / totalCoordinates;
}

/**
 * @brief get the variance of the coordinates high variance indicates a level of exploration 
 * 
 * @param leaf the leaf node in question
 * @return double the score of the exploration metric
 */
double KnnModel::get_exploration_score(TreeNode* leaf) {
    std::vector<double> average(dimensions, 0.0);
    int N = leaf->points.size();

    double maxPotentialVariance = 0;
    for (int i = 0; i < dimensions; i++)
        maxPotentialVariance += pow(leaf->max_bound[i] - leaf->min_bound[i], 2)/4;
    
    maxPotentialVariance /= dimensions;
    
    return (maxPotentialVariance - get_variance(leaf->points))/maxPotentialVariance;
}

void KnnModel::update_prediction(const std::vector<int> &query, double result){
    currentQuery++;

    Point newPoint;
    newPoint.coords = query;
    newPoint.value = result;

    if (root == nullptr) {
        root = new TreeNode();
        root->points.push_back(newPoint);
        root->min_bound = std::vector<int>(dimensions, 0);
        root->max_bound = std::vector<int>(dimensions, dimensionSize-1);
        return;
    }

    // Insert point into tree
    insert_point(root, newPoint);

    if (currentQuery == totalQueries){
        std::vector<int> initialQuery(dimensions, 0);
        updateStateSpace(initialQuery, 0);
    }
}

TreeNode* KnnModel::build_tree(std::vector<Point>& data,
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
    node->left = build_tree(left_pts, min_bound, left_max);

    // Right child bounds
    std::vector<int> right_min = min_bound;
    right_min[best_dim] = node->split_value + 1;
    node->right = build_tree(right_pts, right_min, max_bound);

    return node;
}


// Find leaf for a query
TreeNode* KnnModel::find_leaf(TreeNode* node, const std::vector<int>& query) {
    if (node->is_leaf()) return node;
    if (query[node->split_dimension] <= node->split_value)
        return find_leaf(node->left, query);
    else
        return find_leaf(node->right, query);
}

double KnnModel::get_variance(std::vector<Point>& data){
    std::vector<double> average(dimensions, 0.0);
    int N = data.size();

    for (auto& p : data){
        for (int d = 0; d < dimensions; ++d)
            average[d] += (p.coords[d]/N);
    }

    double avgVariance = 0;

    for (auto& p : data){
        for (int d = 0; d < dimensions; ++d) {
            double diff = p.coords[d] - average[d];
            avgVariance += diff * diff;
        }
    }

    return avgVariance/(N*dimensions);
}

// Insert point and split if necessary
void KnnModel::insert_point(TreeNode* node, const Point& p) {
    if (node->is_leaf()) {
        node->points.push_back(p);

        if (node->points.size() > min_leaf_size && get_variance(node->points) > variance_threshold) {
            std::vector<Point> pts = node->points;
            node->points.clear();
            TreeNode* new_node = build_tree(pts, node->min_bound, node->max_bound);

            // copy split info
            node->split_dimension = new_node->split_dimension;
            node->split_value = new_node->split_value;
            node->left = new_node->left;
            node->right = new_node->right;
            delete new_node;
        }
        return;
    }

    if (p.coords[node->split_dimension] <= node->split_value)
        insert_point(node->left, p);
    else
        insert_point(node->right, p);
}

double KnnModel::predict_coordinate(const std::vector<int>& coordinate){
    int K = 5; // you can make this configurable

    // 1. Find leaf containing the query
    TreeNode* leaf = find_leaf(root, coordinate);

    // 2. Collect candidate points (start with leaf)
    std::vector<Point> candidates = leaf->points;

    if (candidates.empty()) {
        // no data -> return neutral prediction
        return 0.0;
    }

    // 3. Compute distances to all candidates
    struct Neighbor {
        double dist;
        double value;
        bool operator<(const Neighbor& other) const {
            return dist < other.dist; // for sorting
        }
    };

    std::vector<Neighbor> neighbors;
    for (auto& p : candidates) {
        if (p.coords == coordinate)
            return p.value;
        double dist = 0.0;
        for (int d = 0; d < dimensions; d++) {
            double diff = (double)p.coords[d] - coordinate[d];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);

        neighbors.push_back({dist, p.value});
    }

    // 4. Sort by distance
    std::sort(neighbors.begin(), neighbors.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; });

    // 5. Take K nearest (or fewer if not enough points)
    int useK = std::min(K, (int)neighbors.size());

    // 6. Weighted average prediction
    double weightedSum = 0.0;
    double weightTotal = 0.0;

    for (int i = 0; i < useK; i++) {
        double w = (neighbors[i].dist == 0.0) ? 1e9 : 1.0 / neighbors[i].dist; // handle exact match
        weightedSum += w * neighbors[i].value;
        weightTotal += w;
    }

    return weightedSum / weightTotal;
}

void KnnModel::updateStateSpace(std::vector<int>& coords, int idx){
    stateSpace->set(coords, predict_coordinate(coords));

    if (idx == dimensions) return;

    for (int x = 1; x < dimensionSize; x++){
        coords[idx] = x;
        updateStateSpace(coords, idx + 1);
    }
    coords[idx] = 0;
}
#endif