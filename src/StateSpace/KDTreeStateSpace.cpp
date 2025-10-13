#include <algorithm>
#include "KDTreeStateSpace.hpp"

// Constructor
KDTreeStateSpace::KDTreeStateSpace(int dimensions, int dimensionSize)
    : StateSpace(dimensions, dimensionSize), root(nullptr) {
        // Weigh max depth by number of dimensions and dimension size.


        // Can also just choose a fixed max depth. This is a heuristic.
        // Setting maxDepth to -1 means no limit.
        int totalPoints = 1;
        for (int d = 0; d < dimensions; ++d)
            totalPoints *= dimensionSize;

        this->bucketSizeThreshold = static_cast<int>(std::log2(totalPoints)); // log2(N) heuristic

    }

// Destructor
KDTreeStateSpace::~KDTreeStateSpace() {
    destroy(root);
}

// Helper
std::string KDTreeStateSpace::coordsToString(std::vector<int> coords) const {
    std::string key;
    for (int c : coords) {
        key += std::to_string(c) + ",";
    }
    return key;
}

// Recursively free nodes
void KDTreeStateSpace::destroy(TreeNode* node) {
    if (!node) return;

    for (auto* child : node->bucket) {
        delete child;
    }

    destroy(node->left);
    destroy(node->right);
    delete node;
}


// Dimensions accessors
int KDTreeStateSpace::get_dimensions() const {
    return dimensions;
}

int KDTreeStateSpace::get_dimension_size() const {
    return dimensionSize;
}

// Public insert entry point
double KDTreeStateSpace::insert(const std::vector<int>& coords, double value) {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }
    root = insert_recursive(root, coords, value, 0);
    return value;
}

// Recursive insert
KDTreeStateSpace::TreeNode* KDTreeStateSpace::insert_recursive(
    TreeNode* node, const std::vector<int>& coords, double value, int depth)
{
    int axis = depth % this->dimensions;

    // Base case: create new node
    if (!node) {
        TreeNode* newNode = new TreeNode(coords, value, axis);
        newNode->isBucket = true; 
        newNode->bucket.push_back(new TreeNode(coords, value, axis)); // Adding itself to its bucket
        leafNodes.insert(newNode);
        return newNode;
    }

    // If this node is a bucket
    if (node->isBucket) {
        // Update value if coordinate exists
        for (auto* child : node->bucket) {
            if (child->coords == coords) {
                child->value = value;
                return node;
            }
        }

        // Insert new point into bucket
        TreeNode* newNode = new TreeNode(coords, value, axis);
        node->bucket.push_back(newNode);
        leafNodes.insert(newNode);

        // Split only if threshold exceeded
        if (node->bucket.size() > static_cast<size_t>(bucketSizeThreshold)) {

            // 1) Choose split axis by largest range
            int bestAxis = 0;
            int maxRange = -1;
            for (int d = 0; d < dimensions; ++d) {
                int minVal = node->bucket[0]->coords[d];
                int maxVal = node->bucket[0]->coords[d];
                for (auto* child : node->bucket) {
                    minVal = std::min(minVal, child->coords[d]);
                    maxVal = std::max(maxVal, child->coords[d]);
                }
                int range = maxVal - minVal;
                if (range > maxRange) {
                    maxRange = range;
                    bestAxis = d;
                }
            }

            // Sort by best axis and pick median
            std::sort(node->bucket.begin(), node->bucket.end(),
                      [bestAxis](TreeNode* a, TreeNode* b) {
                          return a->coords[bestAxis] < b->coords[bestAxis];
                      });
            size_t medianIndex = node->bucket.size() / 2;
            TreeNode* medianNode = node->bucket[medianIndex];

            // Promote median to current node
            node->coords = medianNode->coords;
            node->value  = medianNode->value;
            node->axis   = bestAxis;
            // Node is no longer a bucket
            node->isBucket = false;
            leafNodes.erase(node);

            // Reinsert all other points
            std::vector<TreeNode*> tmp = std::move(node->bucket);
            node->bucket.clear();
            node->left = node->right = nullptr;

            for (size_t i = 0; i < tmp.size(); ++i) {
                if (i == medianIndex) {
                    delete tmp[i]; // median already promoted
                    continue;
                }
                if (tmp[i]->coords[bestAxis] < node->coords[bestAxis]) {
                    node->left = insert_recursive(node->left, tmp[i]->coords, tmp[i]->value, depth + 1);
                } else {
                    node->right = insert_recursive(node->right, tmp[i]->coords, tmp[i]->value, depth + 1);
                }
                delete tmp[i]; // temporary bucket node no longer needed
            }

            return node;
        }

        return node; // still a bucket, below threshold
    }

    // Standard KD-tree insertion
    if (coords == node->coords) {
        node->value = value;
        return node;
    }

    bool goLeft = (coords[axis] < node->coords[axis]);
    if (!node->left && !node->right && !node->isBucket) leafNodes.erase(node);

    if (goLeft) {
        node->left = insert_recursive(node->left, coords, value, depth + 1);
    } else {
        node->right = insert_recursive(node->right, coords, value, depth + 1);
    }

    // Maintain leafNodes
    if (!node->left && !node->right && !node->isBucket) leafNodes.insert(node);
    else leafNodes.erase(node);

    return node;
}



// Public get entry point
double KDTreeStateSpace::get(const std::vector<int>& coords) const {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }

    try {
        return get_recursive(root, coords, 0);
    } catch (std::out_of_range&) {
        // Return default value if not present
        return 0.0;
    }
}


// Recursive search
double KDTreeStateSpace::get_recursive(TreeNode* node, const std::vector<int>& coords, int depth) const {
    if (!node) throw std::out_of_range("Coordinates not found in KDTree");

    // If this is a bucket node, check all children
    if (node->isBucket) {
        for (auto* child : node->bucket) {
            if (child->coords == coords) {
                return child->value;
            }
        }
        throw std::out_of_range("Coordinates not found in bucket");
    }

    if (coords == node->coords) return node->value;

    int axis = node->axis;
    if (coords[axis] < node->coords[axis]) {
        return get_recursive(node->left, coords, depth + 1);
    } else {
        return get_recursive(node->right, coords, depth + 1);
    }
}


KDTreeStateSpace::TreeNode* KDTreeStateSpace::get_root() {
    return this->root;
}


// Compute squared Euclidean distance (faster than sqrt)
double KDTreeStateSpace::squared_distance(const std::vector<int>& a,
                                          const std::vector<int>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = static_cast<double>(a[i] - b[i]);
        sum += diff * diff;
    }
    return sum;
}

// Public entry point
KDTreeStateSpace::TreeNode*
KDTreeStateSpace::nearest_neighbor(const std::vector<int>& coords) const {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }

    if (!root) {
        // KDTree is empty, return nullptr
        return nullptr;
    }

    TreeNode* nearestNode = nullptr;
    double bestDist = std::numeric_limits<double>::max();

    nearest_recursive(root, coords, 0, nearestNode, bestDist);

    if (!nearestNode) {
        throw std::out_of_range("KDTree is empty");
    }

    // Otherwise return a single-node vector
    return nearestNode;
}


// // Recursive nearest search
void KDTreeStateSpace::nearest_recursive(TreeNode* node,
                                         const std::vector<int>& target,
                                         int depth,
                                         TreeNode*& bestNode,
                                         double& bestDist) const {
    if (!node) return;

    // If this node is a bucket container, stop here — this bucket is the closest
    if (node->isBucket) {
        for (auto* child : node->bucket) {
            double dist = squared_distance(target, child->coords);
            if (dist < bestDist) {
                bestDist = dist;
                bestNode = child;
            }
        }
        return; // no need to go further
    }


    double dist = squared_distance(target, node->coords);
    if (dist < bestDist) {
        bestDist = dist;
        bestNode = node;
    }

    int axis = depth % dimensions;
    TreeNode* near = (target[axis] < node->coords[axis]) ? node->left : node->right;
    TreeNode* far  = (target[axis] < node->coords[axis]) ? node->right : node->left;

    // Explore near side first
    nearest_recursive(near, target, depth + 1, bestNode, bestDist);

    double diff = static_cast<double>(target[axis] - node->coords[axis]);
    if (diff * diff < bestDist) {
        nearest_recursive(far, target, depth + 1, bestNode, bestDist);
    }
}



// std::vector<KDTreeStateSpace::TreeNode*>
// KDTreeStateSpace::nearest_k_neighbors(const std::vector<int>& coords, int k) const {
//     if (coords.size() != static_cast<size_t>(dimensions)) {
//         throw std::invalid_argument("Coordinate dimensionality mismatch");
//     }

//     if (!root) return {};

//     // Min-heap (largest distance at top)
//     using Neighbor = std::pair<double, TreeNode*>;
//     std::priority_queue<
//         Neighbor,
//         std::vector<Neighbor>,
//         std::less<Neighbor>
//     > bestNeighbors;


//     nearest_recursive_k(root, coords, 0, k, bestNeighbors);

//     // Extract nodes from heap
//     std::vector<TreeNode*> results;
//     results.reserve(bestNeighbors.size());
//     while (!bestNeighbors.empty()) {
//         results.push_back(bestNeighbors.top().second);
//         bestNeighbors.pop();
//     }

//     // Reverse so closest is first
//     std::reverse(results.begin(), results.end());
//     return results;
// }

    
// void KDTreeStateSpace::nearest_recursive_k(
//     TreeNode* node,
//     const std::vector<int>& target,
//     int depth,
//     int k,
//     std::priority_queue<
//         std::pair<double, TreeNode*>,
//         std::vector<std::pair<double, TreeNode*>>,
//         std::less<std::pair<double, TreeNode*>>
//     >& bestNeighbors
// ) const {

//     if (!node) return;

//     // If bucket node → check all bucket children
//     if (node->isBucket) {
//         for (auto* child : node->bucket) {
//             double dist = squared_distance(target, child->coords);
//             if (bestNeighbors.size() < (size_t)k) {
//                 bestNeighbors.emplace(dist, child);
//             } else if (dist < bestNeighbors.top().first) {
//                 bestNeighbors.pop();
//                 bestNeighbors.emplace(dist, child);
//             }
//         }
//         return;
//     }

//     // Distance to current node
//     double dist = squared_distance(target, node->coords);
//     if (bestNeighbors.size() < (size_t)k) {
//         bestNeighbors.emplace(dist, node);
//     } else if (dist < bestNeighbors.top().first) {
//         bestNeighbors.pop();
//         bestNeighbors.emplace(dist, node);
//     }

//     int axis = depth % dimensions;
//     TreeNode* near = (target[axis] < node->coords[axis]) ? node->left : node->right;
//     TreeNode* far  = (target[axis] < node->coords[axis]) ? node->right : node->left;

//     // Explore near branch first
//     nearest_recursive_k(near, target, depth + 1, k, bestNeighbors);

//     // Check if far branch might contain closer neighbors
//     double diff = static_cast<double>(target[axis] - node->coords[axis]);
//     if (bestNeighbors.size() < (size_t)k || diff * diff < bestNeighbors.top().first) {
//         nearest_recursive_k(far, target, depth + 1, k, bestNeighbors);
//     }
// }
