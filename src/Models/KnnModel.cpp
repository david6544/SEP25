#if defined(KNN) || defined(TESTING)
#include <vector>
#include <algorithm>

#include "KnnModel.hpp"

// public methods

KnnModel::KnnModel(int dimensions, int dimensionSize, int queries) : Model(dimensions, dimensionSize, queries) {
    
}

void KnnModel::get_explore_leaves(TreeNode* node, std::vector<TreeNode*>& leaves) {
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
            double score = get_exploitation_score(leaf) + 1.4 * get_exploration_score(leaf);
            if (best_score < score){
                best_score = score;
                best_candidate = candidate;
            }
        }
    }
    
    return best_candidate;
}

std::vector<int> KnnModel::get_random_candidate(TreeNode* leaf) {
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
    if (leaf->points.empty()) return 1.0; // completely unexplored

    double avgPred = 0.0;
    for (auto& p : leaf->points) avgPred += p.value;
    avgPred /= leaf->points.size();

    double predVar = 0.0;
    for (auto& p : leaf->points) {
        double diff = p.value - avgPred;
        predVar += diff * diff;
    }
    predVar /= leaf->points.size();

    // Normalize by some max variance
    return predVar / (1.0 + predVar);
}

void KnnModel::update_prediction(const std::vector<int> &query, double result){
    currentQuery++;

    if (root == nullptr) {
        std::vector<Point> data = { Point(query, result) };
        std::vector<int> min_bound(dimensions, 0);
        std::vector<int> max_bound(dimensions, dimensionSize - 1);
        root = build_tree(data, min_bound, max_bound);
        return;
    }

    // Insert point into tree
    insert_point(root, query, result);

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
TreeNode* KnnModel::find_leaf(TreeNode* node, const std::vector<int>& query) {
    if (node->is_leaf()) return node;
    if (query[node->split_dimension] <= node->split_value)
        return find_leaf(node->left, query);
    else
        return find_leaf(node->right, query);
}

double KnnModel::get_variance(std::vector<Point>& data){
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
void KnnModel::insert_point(TreeNode* node, const std::vector<int>& query, double result) {
    if (!node) {
        std::cerr << "insert_point got nullptr node!\n";
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

struct Neighbor {
    double dist2;
    double value;
    bool operator<(const Neighbor& other) const { return dist2 < other.dist2; }
};

void collect_neighbors(TreeNode* node,
                       const std::vector<int>& coord,
                       int K,
                       std::vector<Neighbor>& heap,
                       bool &found_exact) {
    if (!node || found_exact) return;

    if (node->is_leaf()) {
        for (auto& p : node->points) {
            double dist2 = 0.0;
            for (int d = 0; d < (int)coord.size(); ++d) {
                double diff = (double)p.coords[d] - coord[d];
                dist2 += diff * diff;
            }

            if (dist2 == 0.0) {
                // exact match -> store it and signal to stop everything
                heap.clear();
                heap.push_back({0.0, p.value});
                // no need to heapify for single element
                found_exact = true;
                return;
            }

            if ((int)heap.size() < K) {
                heap.push_back({dist2, p.value});
                std::push_heap(heap.begin(), heap.end()); // maintain max-heap
            } else {
                // ensure heap[0] is the largest dist2 (max-heap)
                if (dist2 < heap.front().dist2) {
                    std::pop_heap(heap.begin(), heap.end()); // moves largest to back
                    heap.back() = {dist2, p.value};
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }
        return;
    }

    // Decide which side to explore first
    int dim = node->split_dimension;
    double val = coord[dim];

    TreeNode* first = (val <= node->split_value) ? node->left : node->right;
    TreeNode* second = (val <= node->split_value) ? node->right : node->left;

    // Explore closer side first
    collect_neighbors(first, coord, K, heap, found_exact);
    if (found_exact) return;

    // Decide whether to explore other side
    double diff = val - node->split_value;
    double diff2 = diff * diff;

    // Only explore if closer side didn't already give enough close points
    if ((int)heap.size() < K || diff2 < heap.front().dist2) {
        collect_neighbors(second, coord, K, heap, found_exact);
    }
}


double KnnModel::predict_coordinate(const std::vector<int>& coordinate){
    int K = 5;
    if (!root) return 0.0;

    std::vector<Neighbor> neighbors;
    neighbors.reserve(K);

    bool found_exact = false;
    collect_neighbors(root, coordinate, K, neighbors, found_exact);

    // If exact match found
    if (found_exact && neighbors.size() == 1 && neighbors[0].dist2 == 0.0) {
        return neighbors[0].value;
    }

    if (neighbors.empty()) {
        // no neighbors found -> fallback (choose sensible default)
        return 0.0;
    }

    // Make sure neighbors is a heap (it should already be) and then use them
    std::make_heap(neighbors.begin(), neighbors.end());

    double weightedSum = 0.0, weightTotal = 0.0;
    double epsilon = 1e-6;

    // Note: neighbors is a max-heap, but we can iterate normally
    for (auto& n : neighbors) {
        double w = 1.0 / (std::sqrt(n.dist2) + epsilon); // inverse distance weighting
        weightedSum += w * n.value;
        weightTotal += w;
    }

    return (weightTotal > 0.0) ? (weightedSum / weightTotal) : 0.0;
}



void KnnModel::updateStateSpace(std::vector<int>& coords, int idx){
    if (idx == dimensions) {
        stateSpace->set(coords, predict_coordinate(coords));
        return;
    };

    for (int x = 0; x < dimensionSize; x++){
        coords[idx] = x;
        updateStateSpace(coords, idx + 1);
    }
    coords[idx] = 0;
}
#endif