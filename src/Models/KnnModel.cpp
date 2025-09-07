#if defined(KNN) || defined(TESTING)
#include <vector>
#include <algorithm>

#include "KnnModel.hpp"

// public methods

KnnModel::KnnModel(int dimensions, int dimensionSize, int queries) : Model(dimensions, dimensionSize, queries) {
    
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
        root = KD_Tree::build_tree(data, min_bound, max_bound);
        return;
    }

    // Insert point into tree
    KD_Tree::insert_point(root, query, result);

    if (currentQuery == totalQueries){
        std::vector<int> initialQuery(dimensions, 0);
        updateStateSpace(initialQuery, 0);
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


double KnnModel::get_value_at(const std::vector<int>& coordinate){
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

#endif